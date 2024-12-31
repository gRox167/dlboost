from typing import Dict, Tuple

import einx
import numpy as np
import torch
from deepinv.physics import LinearPhysics, adjoint_function
from dlboost.models import SpatialTransformNetwork
from dlboost.NODEO.Utils import resize_deformation_field
from dlboost.utils.tensor_utils import interpolate
from jaxtyping import Shaped
from mrboost.computation import nufft_2d, nufft_adj_2d
from mrboost.type_utils import ComplexImage3D
from torchopt.linear_solve import solve_cg


class CSM_FixPh(LinearPhysics):
    # python type tuple length 3
    def __init__(self, scale_factor: Tuple[int, int, int] | None = (1, 8, 8)):
        super().__init__()
        self.scale_factor = scale_factor

    def update_parameters(self, csm_kernels: Shaped[ComplexImage3D, "b ch"]):
        if self.scale_factor is not None:
            self._csm = interpolate(
                csm_kernels, scale_factor=self.scale_factor, mode="trilinear"
            )
        else:
            self._csm = csm_kernels

    def A(self, image: Shaped[ComplexImage3D, "b ph"]):
        return einx.dot("b ph ..., b ch ... -> b ph ch ...", image, self._csm)

    def A_adjoint(self, image_multi_ch: Shaped[ComplexImage3D, "b ph ch"], **kwargs):
        return einx.dot(
            "b ph ch ..., b ch ...-> b ph ...", image_multi_ch, self._csm.conj()
        )


class MVF_Dyn(LinearPhysics):
    def __init__(self, size, scale_factor: Tuple[int, int, int] | None = (1, 2, 2)):
        super().__init__()
        self.scale_factor = scale_factor
        self.spatial_transform = SpatialTransformNetwork(size=size, mode="bilinear")

    def update_parameters(self, mvf_kernels: Shaped[torch.Tensor, "b ph v d h w"]):
        # self.ph_to_move = mvf_kernels.shape[0]
        # _mvf = einx.rearrange("ph v d h w -> ph v d h w", mvf_kernels)
        if self.scale_factor is not None:
            self._mvf = resize_deformation_field(mvf_kernels, self.scale_factor)
        else:
            self._mvf = mvf_kernels
        batches, self.ph_to_move, v, d, h, w = self._mvf.shape[-5:]
        self.A_adjoint_func = adjoint_function(
            self._mvf, batches + (self.ph_to_move + 1, d, h, w)
        )

    def A(self, image: Shaped[ComplexImage3D, "b d h w"]):
        image_moving = einx.rearrange(
            "b d h w cmplx-> (b ph) cmplx d h w",
            torch.view_as_real(image),
            ph=self.ph_to_move,
        )
        image_moved = self.spatial_transform(image_moving, self._mvf)
        image_moved = torch.view_as_complex(
            einx.rearrange(
                "(b ph) cmplx d h w -> b ph d h w cmplx",
                image_moved,
                cmplx=2,
                ph=self.ph_to_move,
            )
        )
        return einx.rearrange(
            "b ph d h w, b d h w -> b (ph + 1) d h w",
            image_moved,
            image,
        )

    def A_adjoint(self, y, **kwargs):
        return self.A_adjoint_func(y, **kwargs)


class Repeat(LinearPhysics):
    def __init__(self, pattern: str, repeats: Dict):
        super().__init__()
        self.pattern = pattern
        self.repeats = repeats

    def update_parameters(self, repeats):
        self.repeats = repeats

    def A(self, x):
        return einx.rearrange(self.pattern, x, **self.repeats)

    def A_adjoint(self, y, **kwargs):
        in_, out_ = self.pattern.split("->")
        return einx.sum(out_ + "->" + in_, y, **self.repeats)


class NUFFT(LinearPhysics):
    def __init__(self, nufft_im_size):
        super().__init__()
        self.nufft_im_size = nufft_im_size

    def update_parameters(self, kspace_traj):
        self.kspace_traj = kspace_traj
        # self.A_adjoint = adjoint_function(
        #     self.A,
        # )

    def A(self, image):
        return nufft_2d(
            image,
            self.kspace_traj,
            self.nufft_im_size,
            norm_factor=2 * np.sqrt(np.prod(self.nufft_im_size)),
            # use this line if you have RO oversampling
            # norm_factor=np.sqrt(np.prod(self.nufft_im_size)),
        )

    def A_adjoint(self, kspace_data):
        return nufft_adj_2d(
            kspace_data,
            self.kspace_traj,
            self.nufft_im_size,
            norm_factor=2 * np.sqrt(np.prod(self.nufft_im_size)),
            # use this line if you have RO oversampling
            # norm_factor=np.sqrt(np.prod(self.nufft_im_size)),
        )


class ConjugateGradientFunction(torch.autograd.Function):
    @staticmethod
    def setup_context(ctx, A, b, x_init, max_iter, rtol):
        """
        Args:
            ctx: Context to save tensors for backward pass.
        """
        ctx.A = A
        ctx.max_iter = max_iter
        ctx.rtol = rtol
        ctx.save_for_backward(b)  # Save tensors that require grad

    @staticmethod
    def forward(A, b, x_init, rtol=1e-5, max_iter=100):
        """
        Solves a batch of linear systems A[i]x[i] = b[i] using conjugate gradient.

        Args:
            A: A callable for batched matrix-vector product.
            b: A tensor of right-hand sides, shape (batch_size, n).
            tol: Tolerance for the stopping criterion.
            max_iter: Maximum number of iterations.

        Returns:
            x: Solution tensor, shape (batch_size, n).
        """
        solver = solve_cg(init=x_init)
        x = solver(A, b, rtol=rtol, max_iter=max_iter)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the gradient of the output with respect to the inputs.

        Args:
            ctx: Context with saved tensors from forward.
            grad_output: Gradient of the loss with respect to the output, shape (batch_size, n).

        Returns:
            Gradients with respect to A, b, tol, and max_iter.
        """
        b = ctx.saved_tensors[0]
        grad_b = None
        if ctx.needs_input_grad[1]:  # Gradient w.r.t. b
            solver = solve_cg(init=b)
            grad_b = solver(ctx.A, grad_output, rtol=ctx.rtol, max_iter=ctx.max_iter)
        return (None, grad_b, None, None, None)


if __name__ == "__main__":
    # Example: Define a linear function representing matrix multiplication
    n = 10
    A_matrix = torch.randn(n, n)
    A_matrix = A_matrix @ A_matrix.T  # Make it symmetric positive-definite

    def A_func(x):
        return A_matrix @ x

    b = torch.randn(n)

    # Without functorch
    b.requires_grad_(True)
    x_cg = ConjugateGradientFunction.apply(A_func, b, max_iter=50)
    loss = x_cg.sum()
    loss.backward()
    print("Solution without functorch:", x_cg)
    print("Gradient w.r.t. b:", b.grad)

    # With functorch (vmap)
    batch_size = 3
    b_batch = torch.randn(batch_size, n)

    # Need to ensure A_func can handle batched inputs if we batch the problem
    def batched_A_func(X):
        # X: (batch_size, n)
        return torch.bmm(
            A_matrix.unsqueeze(0).expand(batch_size, -1, -1), X.unsqueeze(-1)
        ).squeeze(-1)

    b_batch.requires_grad_(True)
    x_cg_batch = torch.vmap(
        ConjugateGradientFunction.apply, in_dims=(None, 0, None, None, None)
    )(batched_A_func, b_batch, None, 50, 1e-5)
    loss_batch = x_cg_batch.sum()
    loss_batch.backward()
    print("Solution with functorch:", x_cg_batch)
    print("Gradient w.r.t. b_batch:", b_batch.grad)
