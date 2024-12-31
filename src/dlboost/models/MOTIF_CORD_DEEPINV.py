import deepinv
import torch
from deepinv.optim import RED
from deepinv.physics import LinearPhysics
from torch import nn
from torchopt.linear_solve import solve_cg, solve_normal_cg

from dlboost.models import ComplexUnet, DWUNet
from dlboost.operators import NUFFT, CSM_FixPh, MVF_Dyn


class MR_Forward_Model(LinearPhysics):
    def __init__(
        self,
        MVF_physics=MVF_Dyn((80, 320, 320)),
        CSM_physics=CSM_FixPh((1, 8, 8)),
        NUFFT_physics=NUFFT((320, 320)),
    ):
        super().__init__()
        self.M = MVF_physics
        self.S = CSM_physics
        self.N = NUFFT_physics
        self.forward_model = self.N * self.S * self.M

    def update_parameters(self, mvf_kernels=None, csm_kernels=None, kspace_traj=None):
        if hasattr(self.M, "update_parameters"):
            self.M.update_parameters(mvf_kernels)
        self.S.update_parameters(csm_kernels)
        self.N.update_parameters(kspace_traj)

    def A(self, image):
        return self.forward_model.A(image)

    def A_adjoint(self, kspace_data):
        return self.forward_model.A_adjoint(kspace_data)

    def A_dagger(self, y, **kwargs):
        solver = solve_normal_cg(init=self.A_adjoint(y), **kwargs)
        x_hat = solver(self.A, y)
        return x_hat

    def prox_l2(self, z, y, gamma, **kwargs):
        r"""
        Computes proximal operator of :math:`f(x) = \frac{1}{2}\|Ax-y\|^2`, i.e.,

        .. math::

            \underset{x}{\arg\min} \; \frac{\gamma}{2}\|Ax-y\|^2 + \frac{1}{2}\|x-z\|^2

        :param torch.Tensor y: measurements tensor
        :param torch.Tensor z: signal tensor
        :param float gamma: hyperparameter of the proximal operator
        :return: (torch.Tensor) estimated signal tensor

        """
        b = self.A_adjoint(y) + 1 / gamma * z

        def H(x):
            return self.A_adjoint(self.A(x)) + 1 / gamma * x

        solver = solve_cg(init=z)
        x = solver(H, b, rtol=1e-2)
        print("1 iteration")
        return x


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


class Regularization(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_denoiser = ComplexUnet(
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
            norm_with_given_std=True,
        )

    def forward(self, params, std=None):
        # params = params.clone()
        return self.image_denoiser(params, std=std)


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
