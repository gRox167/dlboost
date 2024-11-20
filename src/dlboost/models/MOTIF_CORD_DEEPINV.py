from typing import Tuple

import deepinv
import einx
import numpy as np
import torch
from deepinv.optim import RED
from deepinv.physics import LinearPhysics, adjoint_function
from jaxtyping import Shaped
from mrboost.computation import nufft_2d, nufft_adj_2d
from mrboost.type_utils import ComplexImage3D
from torch import nn

from dlboost.models import ComplexUnet, DWUNet, SpatialTransformNetwork
from dlboost.NODEO.Utils import resize_deformation_field
from dlboost.utils.tensor_utils import interpolate


class CSM_FixPh(LinearPhysics):
    # python type tuple length 3
    def __init__(self, scale_factor: Tuple[int, int, int] | None = (1, 8, 8)):
        super().__init__()
        self.scale_factor = scale_factor

    def update_parameters(self, csm_kernels: Shaped[ComplexImage3D, "ch"]):
        if self.scale_factor is not None:
            self._csm = interpolate(
                csm_kernels, scale_factor=self.scale_factor, mode="trilinear"
            )
        else:
            self._csm = csm_kernels

    def A(self, image: Shaped[ComplexImage3D, "..."]):
        return einx.dot("... d h w, ch d h w -> ... ch d h w", image, self._csm)

    def A_adjoint(self, image_multi_ch: Shaped[ComplexImage3D, "... ch"], **kwargs):
        return einx.dot(
            "... ch d h w, ch d h w -> ... d h w", image_multi_ch, self._csm
        )


class MVF_Dyn(LinearPhysics):
    def __init__(self, size, scale_factor: Tuple[int, int, int] | None = (1, 2, 2)):
        super().__init__()
        self.scale_factor = scale_factor
        self.spatial_transform = SpatialTransformNetwork(size=size, mode="bilinear")

    def update_parameters(self, mvf_kernels: Shaped[torch.Tensor, "... ph v d h w"]):
        # self.ph_to_move = mvf_kernels.shape[0]
        # _mvf = einx.rearrange("ph v d h w -> ph v d h w", mvf_kernels)
        if self.scale_factor is not None:
            self._mvf = resize_deformation_field(mvf_kernels, self.scale_factor)
        else:
            self._mvf = mvf_kernels
        batches, self.ph_to_move, v, d, h, w = self._mvf.shape[-5:]
        self.A_adjoint = adjoint_function(
            self._mvf, batches + (self.ph_to_move + 1, d, h, w)
        )

    def A(self, image: Shaped[ComplexImage3D, "..."]):
        image_moving = einx.rearrange(
            "b... d h w cmplx-> (b... ph) cmplx d h w",
            torch.view_as_real(image),
            ph=self.ph_to_move,
        )
        image_moved = self.spatial_transform(image_moving, self._mvf)
        image_moved = torch.view_as_complex(
            einx.rearrange(
                "(b... ph) cmplx d h w -> b... ph d h w cmplx",
                image_moved,
                cmplx=2,
                ph=self.ph_to_move,
            )
        )
        return einx.rearrange(
            "b... ph d h w , b... d h w -> b... (ph + 1) d h w",
            image_moved,
            image,
        )


class NUFFT(LinearPhysics):
    def __init__(self, nufft_im_size):
        super().__init__()
        self.nufft_im_size = nufft_im_size
        self.A_adjoint = adjoint_function(
            self.A,
        )

    def update_parameters(self, kspace_traj):
        self.kspace_traj = kspace_traj

    def A(self, image):
        return nufft_2d(
            image,
            self.kspace_traj,
            self.nufft_im_size,
            norm_factor=2 * np.sqrt(np.prod(self.nufft_im_size)),
        )

    def A_adjoint(self, kspace_data):
        return nufft_adj_2d(
            kspace_data,
            self.kspace_traj,
            self.nufft_im_size,
            adjoint=True,
            norm_factor=2 * np.sqrt(np.prod(self.nufft_im_size)),
        )


class MR_Forward_Model(LinearPhysics):
    def __init__(
        self,
        image_size,
        nufft_im_size,
        CSM_physics=CSM_FixPh,
        MVF_physics=MVF_Dyn,
        NUFFT_physics=NUFFT,
    ):
        super().__init__()
        self.M = MVF_physics(image_size)
        self.S = CSM_physics()
        self.N = NUFFT_physics(nufft_im_size)
        self.forward_model = self.N * self.S * self.M

    def update_parameters(self, mvf_kernels, csm_kernels, kspace_traj):
        if hasattr(self.M, "update_parameters"):
            self.M.update_parameters(mvf_kernels)
        self.S.update_parameters(csm_kernels)
        self.N.update_parameters(kspace_traj)

    def A(self, image):
        return self.forward_model.A(image)

    def A_adjoint(self, kspace_data):
        return self.forward_model.A_adjoint(kspace_data)


class ComplexRED(RED):
    def __init__(self):
        self.denoiser = ComplexUnet(
            1,
            1,
            spatial_dims=3,
            conv_net=DWUNet(
                in_channels=2,
                out_channels=2,
                features=(16, 32, 64, 128, 256),
                # features=(32, 64, 128, 256, 512),
                strides=((2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2)),
                kernel_sizes=(
                    (3, 3, 3),
                    (3, 3, 3),
                    (3, 3, 3),
                    (3, 3, 3),
                    (3, 3, 3),
                ),
            ),
            norm_with_given_std=False,
        )
        super().__init__(denoiser=self.denoiser)

    def grad(self, x):
        return x - self.denoiser(x)


# class Regularization(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.image_denoiser = ComplexUnet(
#             1,
#             1,
#             spatial_dims=3,
#             conv_net=DWUNet(
#                 in_channels=2,
#                 out_channels=2,
#                 features=(16, 32, 64, 128, 256),
#                 # features=(32, 64, 128, 256, 512),
#                 strides=((2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2)),
#                 kernel_sizes=(
#                     (3, 3, 3),
#                     (3, 3, 3),
#                     (3, 3, 3),
#                     (3, 3, 3),
#                     (3, 3, 3),
#                 ),
#             ),
#             norm_with_given_std=True,
#         )

#     def forward(self, params, std=None):
#         params = params.clone()
#         return self.image_denoiser(params, std=std)


class Identity_Regularization:
    def __init__(self):
        pass

    def __call__(self, params):
        return params


class MOTIF_CORD(nn.Module):
    def __init__(
        self,
        patch_size: tuple = (16, 320, 320),
        patch_effective_ratio=0.2,
        nufft_im_size: tuple = (320, 320),
        epsilon: float = 1e-2,
        iterations: int = 5,
        gamma_init=0.1,
        tau_init=0.2,
    ):
        super().__init__()
        self.physics = MR_Forward_Model(
            patch_size, nufft_im_size, CSM_FixPh, LinearPhysics, NUFFT
        )
        self.prior = deepinv.optim.TVPrior()
        self.data_fidelity = deepinv.optim.data_fidelity.L2()
        self.model = deepinv.optim.optim_builder(
            iteration="GD",
            max_iter=iterations,
            prior=self.prior,
            data_fidelity=self.data_fidelity,
            params_algo={"stepsize": gamma_init, "lambda": tau_init},
        )

        self.epsilon = epsilon
        self.iterations = iterations

    def forward(
        self,
        kspace_data,
        kspace_traj,
        image_init,
        mvf,
        csm,
        std,
        weights_flag=True,
    ):
        # initialization
        # from monai.visualize import matshow3d
        # matshow3d(
        #     image_init[0, 0, 0:5].abs().cpu().numpy(), cmap="gray", vmin=0, vmax=5
        # )
        # plt.imshow(image_init[0, 0, 40])
        image_init = torch.nan_to_num_(image_init)
        self.ph_num = kspace_data.shape[1]
        image_list = []
        if torch.is_complex(image_init):
            x = image_init
        else:
            x = torch.complex(image_init, torch.zeros_like(image_init))
        image_list.append(image_init.cpu())

        self.forward_model.generate_forward_operators(mvf, csm, kspace_traj)

        x.requires_grad_(True)
        # ic(self.tau)
        # TODO Don know why, but gradient become nan after first iteration.

        # grad_dc_fn = grad(lambda img: self.inner_loss(img, kspace_data))
        # print(std)
        for t in range(self.iterations):
            print("iteration", t, "start")
            # apply forward model to get kspace_data_estimated
            # ic(t, x[0, 0, 0, 0, 0:10])
            # with torch.autograd.detect_anomaly():
            # ic(x[0, 0, 0, 0, 0:10])
            dc_loss = self.inner_loss(x.clone(), kspace_data, weights_flag)
            grad_dc = torch.autograd.grad(dc_loss, x)[0]
            grad_reg = torch.zeros_like(x, dtype=torch.complex64)
            grad_reg[:, :, self.effective_slice] = x[
                :, :, self.effective_slice
            ] - self.regularization(x[:, :, self.effective_slice], std=std)
            updates = -(self.gamma * grad_dc + self.tau[t] * grad_reg)
            # updates = -self.gamma * grad_dc
            x = x.add(updates)
            # ic("after add", x[0, 0, 0, 0, 0:10])
            image_list.append(x.clone().detach().cpu())
            print(f"t: {t}, loss: {dc_loss}")

        return x, image_list

    def image_init(self, image_multi_ch, csm):
        image_init = torch.sum(image_multi_ch * csm.conj(), dim=2)
        return image_init

    def inner_loss(self, x, kspace_data, weights_flag):
        # ic(x[0, 0, 0, 0, 0:10])
        kspace_data_estimated = self.forward_model(x)
        # ic(kspace_data_estimated[0, 0, 0, 0, 0:10])
        if weights_flag:
            kspace_data_estimated_detatched = (
                kspace_data_estimated[:, :, :, self.effective_slice].detach().abs()
            )
            norm_factor = kspace_data_estimated_detatched.max()
            weights = 1 / (kspace_data_estimated_detatched / norm_factor + self.epsilon)
        else:
            weights = 1

        # ic(diff[0, 0, 0, 0, 0:10])
        loss_dc = self.loss_fn(
            torch.view_as_real(
                weights * kspace_data_estimated[:, :, :, self.effective_slice]
            ),
            torch.view_as_real(weights * kspace_data[:, :, :, self.effective_slice]),
        )
        return loss_dc
