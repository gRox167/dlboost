from typing import Dict, Tuple

import einx
import numpy as np
import torch
from deepinv.optim.utils import least_squares
from deepinv.physics import LinearPhysics, adjoint_function
from dlboost.models import SpatialTransformNetwork
from dlboost.NODEO.Utils import resize_deformation_field
from dlboost.utils.tensor_utils import interpolate
from jaxtyping import Shaped
from mrboost.computation import (
    fft_1D,
    ifft_1D,
    kspace_point_to_radial_spokes,
    nufft_2d,
    nufft_3d,
    nufft_adj_2d,
    nufft_adj_3d,
    radial_spokes_to_kspace_point,
)
from mrboost.type_utils import ComplexImage3D, KspaceSpokesData, KspaceSpokesTraj
from torchopt.linear_solve import solve_cg


class CSM(LinearPhysics):
    def __init__(
        self,
        scale_factor: Tuple[int, int, int] | None = (1, 8, 8),
        # compute_device="cuda",
        # storage_device="cpu",
    ):
        super().__init__()
        self.scale_factor = scale_factor
        # self.compute_device = compute_device
        # self.storage_device = storage_device

    def update_parameters(self, csm_kernels: Shaped[ComplexImage3D, "b ch"]):
        if self.scale_factor is not None:
            self._csm = interpolate(
                csm_kernels, scale_factor=self.scale_factor, mode="trilinear"
            )
        else:
            self._csm = csm_kernels

    def A(self, image: Shaped[ComplexImage3D, "b"]):
        return einx.dot("b ..., b ch ... -> b ch ...", image, self._csm)

    def A_adjoint(self, image_multi_ch: Shaped[ComplexImage3D, "b ch"], **kwargs):
        return einx.sum(
            "b [ch] ...", image_multi_ch * self._csm.conj()
        )  # .to(self.compute_device)


class CSM_FixPh(LinearPhysics):
    def __init__(
        self,
        scale_factor: Tuple[int, int, int] | None = (1, 8, 8),
        # compute_device="cuda",
        # storage_device="cpu",
    ):
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
            b, ph, v, d, h, w = mvf_kernels.shape
            mvf_kernels = einx.rearrange("b ph v d h w -> (b ph) v d h w", mvf_kernels)
            mvf_kernels = resize_deformation_field(mvf_kernels, self.scale_factor)
            self._mvf = einx.rearrange(
                "(b ph) v d h w -> b ph v d h w", mvf_kernels, ph=ph
            )
        else:
            self._mvf = mvf_kernels
        batches, self.ph_to_move, v, d, h, w = self._mvf.shape
        self.A_adjoint_func = adjoint_function(
            self.A,
            # (batches, self.ph_to_move + 1, d, h, w),
            (batches, d, h, w),
            device=self._mvf.device,
            dtype=torch.complex64,
        )
        if self.spatial_transform.grid.device != self._mvf.device:
            self.spatial_transform.grid = self.spatial_transform.grid.to(
                self._mvf.device
            )

    def A(self, image: Shaped[ComplexImage3D, "b d h w"]):
        image_moving = einx.rearrange(
            "b d h w cmplx-> (b ph) cmplx d h w",
            torch.view_as_real(image),
            ph=self.ph_to_move,
        )
        mvf = einx.rearrange(
            "b ph v d h w -> (b ph) v d h w", self._mvf, ph=self.ph_to_move
        )
        image_moved = self.spatial_transform(image_moving, mvf)
        image_moved = torch.view_as_complex(
            einx.rearrange(
                "(b ph) cmplx d h w -> b ph d h w cmplx",
                image_moved,
                cmplx=2,
                ph=self.ph_to_move,
            ).contiguous()
        )
        return einx.rearrange(
            "b ph d h w, b d h w -> b (ph + 1) d h w",
            image_moved,
            image,
        )

    def A_adjoint(self, y, **kwargs):
        """Calculate adjoint operator using autograd"""
        return self.A_adjoint_func(y, **kwargs)
        # Create a dummy input with requires_grad=True
        # batch_size = y.shape[0]
        # d, h, w = y.shape[2], y.shape[3], y.shape[4]
        # x = torch.zeros(
        #     batch_size, d, h, w, dtype=y.dtype, device=y.device, requires_grad=True
        # )

        # # Apply the forward operator
        # Ax = self.A(x)

        # # Compute gradient of inner product with respect to x
        # grad = torch.autograd.grad(
        #     Ax, x, create_graph=kwargs.get("create_graph", True)
        # )[0]

        # return grad


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


class NUFFT_3D(LinearPhysics):
    def __init__(self, nufft_im_size):
        super().__init__()
        self.nufft_im_size = nufft_im_size

    def update_parameters(self, kspace_traj):
        self.kspace_traj = kspace_traj

    def A(self, image):
        output = nufft_3d(
            image,
            radial_spokes_to_kspace_point(self.kspace_traj),
            self.nufft_im_size,
            norm_factor=2 * np.sqrt(np.prod(self.nufft_im_size)),
            # use this line if you have RO oversampling
            # norm_factor=np.sqrt(np.prod(self.nufft_im_size)),
        )
        return kspace_point_to_radial_spokes(output, self.kspace_traj.shape[-1])

    def A_adjoint(self, kspace_data: Shaped[KspaceSpokesData, "..."]):
        return nufft_adj_3d(
            radial_spokes_to_kspace_point(kspace_data),
            radial_spokes_to_kspace_point(self.kspace_traj),
            self.nufft_im_size,
            norm_factor=2 * np.sqrt(np.prod(self.nufft_im_size)),
            # use this line if you have RO oversampling
            # norm_factor=np.sqrt(np.prod(self.nufft_im_size)),
        )


class NUFFT_2D_FFT_1D(LinearPhysics):
    def __init__(self):  # , compute_device, storage_device):
        super().__init__()
        # self.compute_device = compute_device
        # self.storage_device = storage_device

    def update_parameters(self, kspace_traj, nufft_im_size=None):
        if nufft_im_size is not None:
            self.nufft_im_size = nufft_im_size
        if kspace_traj is not None:
            self.spoke_len = kspace_traj.shape[-1]
            self.kspace_spoke_traj = kspace_traj
            self.kspace_traj = radial_spokes_to_kspace_point(kspace_traj)

    def A(
        self,
        image: Shaped[ComplexImage3D, "*b ch"],
    ) -> Shaped[KspaceSpokesData, "*b ch kz"]:
        _kxkyz = nufft_2d(
            image,
            self.kspace_traj,
            self.nufft_im_size,
            norm_factor=2 * np.sqrt(np.prod(self.nufft_im_size)),
            # use this line if you have RO oversampling
            # norm_factor=np.sqrt(np.prod(self.nufft_im_size)),
        )
        kspace_data = fft_1D(_kxkyz, dim=-2)
        return kspace_point_to_radial_spokes(kspace_data, self.spoke_len)

    def A_adjoint(
        self,
        kspace_data: Shaped[KspaceSpokesData, "*b ch kz"],
    ) -> Shaped[ComplexImage3D, "*b ch"]:
        kspace_data = radial_spokes_to_kspace_point(kspace_data)
        _kxkyz = ifft_1D(kspace_data, dim=-2)

        return nufft_adj_2d(
            _kxkyz,
            self.kspace_traj,
            self.nufft_im_size,
            norm_factor=2 * np.sqrt(np.prod(self.nufft_im_size)),
            # use this line if you have RO oversampling
            # norm_factor=np.sqrt(np.prod(self.nufft_im_size)),
        )


class KspaceMask_kz(LinearPhysics):
    def __init__(self):
        super().__init__()
        self.kspace_mask = None

    def update_parameters(self, kspace_mask):
        self.kspace_mask = kspace_mask

    def A(self, kspace_data: Shaped[KspaceSpokesData, "... kz"]):
        return einx.dot(
            "... kz sp len, kz -> ... kz sp len", kspace_data, self.kspace_mask
        )

    def A_adjoint(self, kspace_data: Shaped[KspaceSpokesData, "... kz"]):
        return einx.dot(
            "... kz sp len, kz -> ... kz sp len", kspace_data, self.kspace_mask
        )


class KspaceMask(LinearPhysics):
    def __init__(self):
        super().__init__()
        self.kspace_mask = None

    def update_parameters(self, kspace_mask):
        self.kspace_mask = kspace_mask

    def A(self, kspace_data):
        return kspace_data * self.kspace_mask

    def A_adjoint(self, kspace_data):
        return kspace_data * self.kspace_mask


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


class RadialVibePhysics(LinearPhysics):
    def __init__(
        self,
        csm_operator=CSM(None),
        nufft_operator=NUFFT_2D_FFT_1D(),
        kspace_mask_operator=KspaceMask(),
    ):
        super().__init__()
        self.csm_operator = csm_operator
        self.nufft_operator = nufft_operator
        self.kspace_mask_operator = kspace_mask_operator

    def update_parameters(
        self,
        ktraj: Shaped[KspaceSpokesTraj, "b"] | None = None,
        csm: Shaped[ComplexImage3D, "b ch"] | None = None,
        kspace_mask: Shaped[KspaceSpokesData, "b ch kz"] | None = None,
        nufft_im_size: tuple[int, int] | None = None,
    ):
        if ktraj is not None or nufft_im_size is not None:
            self.nufft_operator.update_parameters(ktraj, nufft_im_size)

        if csm is not None:
            self.csm_operator.update_parameters(csm)

        if kspace_mask is not None:
            self.kspace_mask_operator.update_parameters(kspace_mask)

    def A(self, x: Shaped[ComplexImage3D, "b"]) -> Shaped[KspaceSpokesData, "b ch kz"]:
        # Apply coil sensitivity operator
        x_coils = self.csm_operator.A(x)

        # Apply NUFFT and 1D FFT
        kspace_data = self.nufft_operator.A(x_coils)

        # Apply kspace mask operator
        kspace_data = self.kspace_mask_operator.A(kspace_data)

        return kspace_data

    def A_adjoint(
        self, y: Shaped[KspaceSpokesData, "b ch kz"]
    ) -> Shaped[ComplexImage3D, "b"]:
        # Apply adjoint of kspace mask operator
        y_masked = self.kspace_mask_operator.A_adjoint(y)

        # Apply adjoint of NUFFT and 1D FFT
        image_coils = self.nufft_operator.A_adjoint(y_masked)

        # Apply adjoint of coil sensitivity operator
        image = self.csm_operator.A_adjoint(image_coils)

        return image

    def A_dagger(self, y, x_init=None, **kwargs):
        with torch.no_grad():
            # if x_init in kwargs, use it as the initial guess
            if x_init is not None:
                init = x_init
            else:
                init = self.A_adjoint(y)
            x_hat = least_squares(
                self.A, self.A_adjoint, y, init=init, tol=1e-3, **kwargs
            )
            # solver = solve_normal_cg(init=init, **kwargs)
            # x_hat = solver(self.A, y)
        return x_hat


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
