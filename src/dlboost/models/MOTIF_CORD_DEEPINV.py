from types import MethodType

import deepinv as dinv
import torch
from deepinv.optim.data_fidelity import L2
from deepinv.optim.optimizers import create_iterator
from deepinv.physics import LinearPhysics
from deepinv.unfolded import BaseUnfold
from mrboost.density_compensation.area_based_radial import (
    area_based_radial_density_compensation,
)

from dlboost.iterators.gradient_descent import GDIteration
from dlboost.loss.weightedL2 import WeightedL2Distance
from dlboost.models import ComplexUnet, DWUNetSmall
from dlboost.operators.MRI import (
    NUFFT,
    CSM_FixPh,
    MVF_Dyn,
    Repeat,
)
from dlboost.priors.pnp import PnP
from dlboost.priors.red import RED


class MR_Forward_Model(LinearPhysics):
    def __init__(
        self,
        MVF_physics: MVF_Dyn | Repeat = MVF_Dyn((80, 320, 320)),
        CSM_physics=CSM_FixPh((1, 8, 8)),
        NUFFT_physics=NUFFT((320, 320)),
        KspaceMask_physics=None,
        max_iter=100,
        tol=1e-2,
        device=torch.device("cuda:0"),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.M = MVF_physics
        self.S = CSM_physics
        self.N = NUFFT_physics
        if KspaceMask_physics is None:
            self.forward_model = self.N * self.S * self.M
        else:
            self.MASK = KspaceMask_physics
            self.forward_model = self.MASK * self.N * self.S * self.M

        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.weight = None

    def update_parameters(
        self,
        mvf_kernels=None,
        csm_kernels=None,
        kspace_traj=None,
        kspace_mask=None,
        nufft_im_size=None,
    ):
        if hasattr(self.M, "update_parameters") and mvf_kernels is not None:
            self.M.update_parameters(mvf_kernels.to(self.device))
        if hasattr(self.S, "update_parameters") and csm_kernels is not None:
            self.S.update_parameters(csm_kernels.to(self.device))
        if hasattr(self, "MASK") and kspace_mask is not None:
            self.MASK.update_parameters(kspace_mask.to(self.device))
        if hasattr(self.N, "update_parameters") and (
            nufft_im_size is not None or kspace_traj is not None
        ):
            self.N.update_parameters(kspace_traj.to(self.device), nufft_im_size)

    def A(self, image):
        input_device = image.device
        image = image.to(self.device)
        return self.forward_model.A(image).to(input_device)

    def A_adjoint(self, kspace_data):
        input_device = kspace_data.device
        kspace_data = kspace_data.to(self.device)
        return self.forward_model.A_adjoint(kspace_data).to(input_device)

    def A_A_adjoint(self, y, **kwargs):
        input_device = y.device
        y = y.to(self.device)
        return super().A_A_adjoint(y, **kwargs).to(input_device)

    def A_adjoint_A(self, x, **kwargs):
        input_device = x.device
        x = x.to(self.device)
        return super().A_adjoint_A(x, **kwargs).to(input_device)

    def A_dagger(
        self, y, solver="CG", max_iter=None, tol=None, verbose=False, **kwargs
    ):
        input_device = y.device
        y = y.to(self.device)
        return (
            super()
            .A_dagger(y, solver, max_iter, tol, verbose, **kwargs)
            .to(input_device)
        )

    def prox_l2(
        self, z, y, gamma, solver="CG", max_iter=None, tol=None, verbose=False, **kwargs
    ):
        input_device = y.device
        z = z.to(self.device)
        y = y.to(self.device)
        return (
            super()
            .prox_l2(z, y, gamma, solver, max_iter, tol, verbose, **kwargs)
            .to(input_device)
        )


dwunet = ComplexUnet(
    1,
    1,
    spatial_dims=3,
    # conv_net=DWUNet(
    conv_net=DWUNetSmall(
        in_channels=2,
        out_channels=2,
        features=(16, 32, 64, 128, 256),
        # features=(32, 64, 128, 256, 512),
        strides=((2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2)),
        kernel_sizes=(
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
        ),
    ),
    input_append_channel=None,
    norm_with_given_std=True,
)


def custom_init(y, physics):
    x_init = physics.A_adjoint(y * physics.weight)
    return {"est": (x_init, torch.zeros_like(x_init))}
    # return {"est": (torch.zeros_like(x_init), torch.zeros_like(x_init))}


class RARE_Phase2Phase_RED(BaseUnfold):
    def __init__(
        self,
        max_iter,
        params_algo,
        trainable_params=[],
        denoiser=ComplexUnet(
            5,
            5,
            spatial_dims=3,
            conv_net=DWUNetSmall(
                in_channels=10,
                out_channels=10,
                features=(
                    16,
                    32,
                    64,
                    128,
                    256,
                ),
                strides=((2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2)),
                kernel_sizes=(
                    (3, 3, 3),
                    (3, 3, 3),
                    (3, 3, 3),
                    (3, 3, 3),
                    (3, 3, 3),
                ),
            ),
            input_append_channel=None,
            norm_with_given_std=True,
        ),
    ):
        data_fidelity = L2()
        # data_fidelity.d = WeightedL2Distance()
        # Unrolled optimization algorithm parameters
        _params_algo = {k: [v] * max_iter for k, v in params_algo.items()}
        prior = RED(denoiser)
        # prior = None
        iterator = create_iterator(
            GDIteration(line_search=False),
            prior=prior,
        )
        super().__init__(
            iterator,
            max_iter=max_iter,
            trainable_params=trainable_params,
            has_cost=iterator.has_cost,
            data_fidelity=data_fidelity,
            prior=prior,
            params_algo=_params_algo,
            custom_init=custom_init,
            verbose=True,
        )

    def forward(self, y, physics, x_gt=None, compute_metrics=False, **kwargs):
        return super().forward(y, physics, x_gt, compute_metrics, **kwargs)


class SD_RED(BaseUnfold):
    def __init__(
        self,
        max_iter,
        params_algo,
        trainable_params=["stepsize", "lambda"],
    ):
        denoiser = dwunet
        data_fidelity = L2()
        # data_fidelity.d = WeightedL2Distance()
        # Unrolled optimization algorithm parameters
        _params_algo = {k: [v] * max_iter for k, v in params_algo.items()}
        prior = RED(denoiser)
        # prior = None
        iterator = create_iterator(
            GDIteration(line_search=False),
            prior=prior,
        )
        super().__init__(
            iterator,
            max_iter=max_iter,
            trainable_params=trainable_params,
            has_cost=iterator.has_cost,
            data_fidelity=data_fidelity,
            prior=prior,
            params_algo=_params_algo,
            custom_init=custom_init,
            verbose=True,
        )

    def forward(self, y, physics, x_gt=None, compute_metrics=False, **kwargs):
        return super().forward(y, physics, x_gt, compute_metrics, **kwargs)


class ADMM(BaseUnfold):
    def __init__(self, max_iter, params_algo, pretrained_params_algo=False):
        denoiser = dwunet
        data_fidelity = L2()
        data_fidelity.d = WeightedL2Distance()
        # Unrolled optimization algorithm parameters
        lamb = params_algo["lambda"] * max_iter  # regularization parameter
        stepsize = params_algo["stepsize"] * max_iter  # step sizes.
        sigma_denoiser = params_algo["g_param"] * max_iter  # denoiser parameters
        params_algo = {  # wrap all the restoration parameters in a 'params_algo' dictionary
            "stepsize": stepsize,
            "g_param": sigma_denoiser,
            "lambda": lamb,
        }
        trainable_params = ["stepsize", "g_param", "lambda"]
        prior = PnP(denoiser)
        iterator = create_iterator(
            "ADMM",
            prior=prior,
        )
        super().__init__(
            iterator,
            max_iter=max_iter,
            trainable_params=trainable_params,
            has_cost=iterator.has_cost,
            data_fidelity=data_fidelity,
            prior=prior,
            params_algo=params_algo,
            custom_init=custom_init,
            verbose=True,
        )


def create_unfold_model(
    iteration,
    max_iter,
    data_fidelity,
    params_algo,
    pretrained_params_algo=False,
    physics_device=torch.device("cuda:0"),
    denoiser_device=torch.device("cuda:1"),
):
    def custom_init(y, physics):
        w = (
            1
            / 4
            * area_based_radial_density_compensation(physics.N.kspace_spoke_traj[0, 0])
        ).to(y.device)
        x_init = physics.A_adjoint(y * w)
        return {"est": (x_init, torch.zeros_like(x_init))}
        # return {"est": (torch.zeros_like(x_init), torch.zeros_like(x_init))}

    if iteration == "ADMM":
        return dinv.unfolded.unfolded_builder(
            iteration="ADMM",
            prior=PnP(denoiser=denoiser),
            max_iter=max_iter,
            trainable_params=params_algo if pretrained_params_algo else [],
            device=denoiser_device,
            data_fidelity=data_fidelity,
            params_algo=params_algo,
            custom_init=custom_init,
            verbose=True,
        )
    elif iteration == "RED":
        return dinv.unfolded.unfolded_builder(
            iteration=GDIteration,
            prior=RED(denoiser=denoiser),
            max_iter=max_iter,
            trainable_params=params_algo if pretrained_params_algo else [],
            device=denoiser_device,
            data_fidelity=data_fidelity,
            params_algo=params_algo,
            custom_init=custom_init,
            verbose=True,
        )


def create_reconstructor(
    denoiser_device=torch.device("cuda:1"),
):
    denoiser = ComplexUnet(
        1,
        1,
        spatial_dims=3,
        # conv_net=DWUNet(
        conv_net=DWUNetSmall(
            in_channels=2,
            out_channels=2,
            features=(16, 32, 64, 128, 256),
            # features=(32, 64, 128, 256, 512),
            strides=((2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2)),
            kernel_sizes=(
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
            ),
        ),
        input_append_channel=None,
        norm_with_given_std=True,
    ).to(denoiser_device)

    reconstructor = dinv.models.ArtifactRemoval(denoiser, mode="adjoint")

    def backbone_inference(self, tensor_in, physics, y):
        return self.backbone_net(tensor_in.unsqueeze(1), 1).squeeze(1)

    reconstructor.backbone_inference = MethodType(backbone_inference, reconstructor)
    return reconstructor
