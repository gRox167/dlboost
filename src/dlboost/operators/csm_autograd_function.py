import einx
import torch
from deepinv.physics import LinearPhysics
from dlboost.utils.tensor_utils import interpolate
from torch import Tensor


class CSMFunctionNoSave(torch.autograd.Function):
    """
    Custom autograd function for applying the coil sensitivity multiplication
    (forward operator) without saving coil sensitivity maps.
    """

    @staticmethod
    def forward(
        ctx, x: Tensor, csm: Tensor, operator: LinearPhysics, channel_index=None
    ):
        # Do not save csm (reduce memory)
        if channel_index is not None:
            _csm = csm[..., channel_index : channel_index + 1, :, :, :]
        else:
            _csm = csm
        if operator.scale_factor is not None:
            _csm = interpolate(
                _csm,
                scale_factor=operator.scale_factor,
                mode="trilinear",
                align_corners=True,
            )
        # compute the forward pass
        out = einx.dot(operator.dot_descriptor, x, _csm)
        ctx.operator = operator
        ctx.channel_index = channel_index
        # Only save x if any gradients (for grad_csm) are needed.
        if ctx.needs_input_grad[1]:
            ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors if ctx.saved_tensors else (None,)
        operator = ctx.operator
        channel_index = ctx.channel_index

        # Recompute coil sensitivity on the fly using operator._csm
        csm = operator._csm
        if channel_index is not None:
            csm = csm[..., channel_index : channel_index + 1, :, :, :]
        if operator.scale_factor is not None:
            csm = interpolate(
                csm,
                scale_factor=operator.scale_factor,
                mode="trilinear",
                align_corners=True,
            )

        # grad for x is computed from the adjoint operation
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = einx.dot(
                operator.adjoint_dot_descriptor,
                grad_output,
                csm.conj(),
                **operator.kwargs,
            )

        grad_csm = None
        if ctx.needs_input_grad[1]:
            # Here we need x to compute grad for csm.
            # Since the operation is linear:
            #    y = einx.dot(descriptor, x, _csm)
            # the derivative w.r.t. _csm is given by:
            #    grad_csm = grad_output * x.conj(), with appropriate unsqueezing/summing.
            x_saved = x  # x was saved if needed
            x_c = x_saved.conj().unsqueeze(x_saved.dim() - 3)
            grad_csm_full = grad_output * x_c
            if x_saved.dim() > 4:
                sum_dims = tuple(range(1, x_saved.dim() - 3))
                grad_csm = grad_csm_full.sum(dim=sum_dims)
            else:
                grad_csm = grad_csm_full

        # No gradients for operator and channel_index
        return grad_x, grad_csm, None, None


class CSMAdjointFunctionNoSave(torch.autograd.Function):
    """
    Custom autograd function for applying the adjoint operator without saving coil sensitivity maps.
    """

    @staticmethod
    def forward(
        ctx, y: Tensor, csm: Tensor, operator: LinearPhysics, channel_index=None
    ):
        if channel_index is not None:
            _csm = csm[..., channel_index : channel_index + 1, :, :, :]
        else:
            _csm = csm
        if operator.scale_factor is not None:
            _csm = interpolate(
                _csm,
                scale_factor=operator.scale_factor,
                mode="trilinear",
                align_corners=True,
            )
        out = einx.dot(
            operator.adjoint_dot_descriptor, y, _csm.conj(), **operator.kwargs
        )
        ctx.operator = operator
        ctx.channel_index = channel_index
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            ctx.save_for_backward(y)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors if ctx.saved_tensors else (None,)
        operator = ctx.operator
        channel_index = ctx.channel_index

        # Recompute coil sensitivity on the fly
        csm = operator._csm
        if channel_index is not None:
            csm = csm[..., channel_index : channel_index + 1, :, :, :]
        if operator.scale_factor is not None:
            csm = interpolate(
                csm,
                scale_factor=operator.scale_factor,
                mode="trilinear",
                align_corners=True,
            )

        grad_y = None
        if ctx.needs_input_grad[0]:
            # For the adjoint:
            grad_y = einx.dot(operator.dot_descriptor, grad_output, csm)
        grad_csm = None
        if ctx.needs_input_grad[1]:
            y_saved = y
            # derivative w.r.t. csm (using conjugation rules)
            y_c = y_saved.conj().unsqueeze(y_saved.dim() - 3)
            grad_csm_full = grad_output * y_c
            if y_saved.dim() > 4:
                sum_dims = tuple(range(1, y_saved.dim() - 3))
                grad_csm = grad_csm_full.sum(dim=sum_dims)
            else:
                grad_csm = grad_csm_full
            # if channel_index is not None:
            #     temp = torch.zeros_like(operator._csm)
            #     temp[..., channel_index : channel_index + 1, :, :, :] = grad_csm
            #     grad_csm = temp

        return grad_y, grad_csm, None, None
