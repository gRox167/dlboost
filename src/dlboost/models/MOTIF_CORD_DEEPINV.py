import deepinv
import torch
from deepinv.optim import RED, OptimIterator, Prior
from deepinv.optim.optimizers import fStep, gStep
from deepinv.physics import LinearPhysics
from torch import nn
from torchopt.linear_solve import solve_cg, solve_normal_cg

from dlboost.models import ComplexUnet, DWUNet
from dlboost.operators.MRI import (
    NUFFT,
    NUFFT_2D_FFT_1D,
    CSM_FixPh,
    KspaceMask_kz,
    MVF_Dyn,
    Repeat,
)


class MR_Forward_Model(LinearPhysics):
    def __init__(
        self,
        MVF_physics: MVF_Dyn | Repeat = MVF_Dyn((80, 320, 320)),
        CSM_physics=CSM_FixPh((1, 8, 8)),
        NUFFT_physics=NUFFT((320, 320)),
        KspaceMask_physics=None,
    ):
        super().__init__()
        self.M = MVF_physics
        self.S = CSM_physics
        self.N = NUFFT_physics
        if KspaceMask_physics is None:
            self.forward_model = self.N * self.S * self.M
        else:
            self.MASK = KspaceMask_physics
            self.forward_model = self.MASK * self.N * self.S * self.M

    def update_parameters(
        self, mvf_kernels=None, csm_kernels=None, kspace_traj=None, kspace_mask=None
    ):
        if hasattr(self.M, "update_parameters"):
            self.M.update_parameters(mvf_kernels)
        self.S.update_parameters(csm_kernels)
        self.N.update_parameters(kspace_traj)
        if hasattr(self, "MASK"):
            self.MASK.update_parameters(kspace_mask)

    def A(self, image):
        return self.forward_model.A(image)

    def A_adjoint(self, kspace_data):
        return self.forward_model.A_adjoint(kspace_data)

    def A_dagger(self, y, x_init=None, **kwargs):
        # if x_init in kwargs, use it as the initial guess
        if x_init is not None:
            init = x_init
        else:
            init = self.A_adjoint(y)
        solver = solve_normal_cg(init=init, **kwargs)
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
        return x


class Identity_Regularization:
    def __init__(self):
        pass

    def __call__(self, params):
        return params


class ComplexRED(Prior):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.denoiser = ComplexUnet(
            in_channels=1,
            out_channels=1,
            spatial_dims=3,
            conv_net=DWUNet(
                in_channels=2,
                out_channels=2,
                spatial_dims=3,
                strides=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
                kernel_sizes=(
                    (3, 3, 3),
                    (3, 3, 3),
                    (3, 3, 3),
                    (3, 3, 3),
                    (3, 3, 3),
                ),
                features=(16, 32, 64, 128, 256),
            ),
            norm_with_given_std=True,
        )

    def grad(self, x, sigma_denoiser):
        return x - self.denoiser(x.unsqueeze(1), std=1).squeeze(1)


class fStep_FP_RED(fStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, cur_data_fidelity, cur_params, y, physics):
        # return x - cur_params["stepsize"] * physics.A_dagger(y, x_init=x, rtol=1e-2)
        # return physics.A_dagger(y, x_init=x, rtol=1e-2)
        return physics.prox_l2(x, y, cur_params["stepsize"])


class gStep_FP_RED(gStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, cur_prior, cur_params):
        grad = cur_params["lambda"] * cur_prior.grad(x, cur_params["g_param"])
        return x - grad


class MOTIF_FP_RED(OptimIterator):
    def __init__(
        self,
        physics: LinearPhysics = MR_Forward_Model(
            # Repeat("b d h w -> b ph d h w", dict(ph=5)),
            MVF_Dyn((80, 320, 320), (1, 4, 4)),
            CSM_FixPh(None),
            NUFFT_2D_FFT_1D((320, 320)),
            KspaceMask_kz(),
        ),
        iterations: int = 5,
        stepsize=1.0,
        lambda_=0.2,
    ):
        super().__init__()
        self.physics = physics
        self.prior = ComplexRED()
        self.data_fidelity = deepinv.optim.data_fidelity.L2()
        self.g_step = gStep_FP_RED()
        self.f_step = fStep_FP_RED()
        self.cur_params = {
            "stepsize": 1.0,
            "lambda": lambda_,
            "g_param": None,
        }
        self.iterations = iterations

    def forward(
        self,
        kspace_data,
        kspace_traj,
        kspace_mask,
        mvf,
        csm,
        image_init=None,
        iterations=None,
        stepsize=None,
        lambda_=None,
    ):
        self.cur_params["stepsize"] = (
            stepsize if stepsize is not None else self.cur_params["stepsize"]
        )
        self.cur_params["lambda"] = (
            lambda_ if lambda_ is not None else self.cur_params["lambda"]
        )
        t = iterations if iterations is not None else self.iterations

        self.physics.update_parameters(
            mvf_kernels=mvf,
            csm_kernels=csm,
            kspace_traj=kspace_traj,
            kspace_mask=kspace_mask,
        )
        y = kspace_data
        x = self.physics.A_dagger(y, rtol=1e-2) if image_init is None else image_init
        x_list = [x.detach().clone()]
        for _ in range(t):
            x = self.g_step(x, self.prior, self.cur_params)
            x = self.f_step(x, self.data_fidelity, self.cur_params, y, self.physics)
            x_list.append(x.detach().clone())

        return x, x_list
