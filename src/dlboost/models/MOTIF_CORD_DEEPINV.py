from types import MethodType

import deepinv as dinv
import torch
from deepinv.optim.data_fidelity import L2
from deepinv.optim.optimizers import create_iterator
from deepinv.optim.utils import least_squares
from deepinv.physics import LinearPhysics
from deepinv.unfolded import BaseUnfold
from mrboost.density_compensation.area_based_radial import (
    area_based_radial_density_compensation,
)

from dlboost.iterators.gradient_descent import GDIteration

# from dlboost.loss.data_fidelity import L2
from dlboost.loss.weightedL2 import WeightedL2Distance
from dlboost.models import ComplexUnet, DWUNetSmall
from dlboost.operators.MRI import (
    CSM,
    NUFFT,
    NUFFT_2D_FFT_1D,
    NUFFT_3D,
    CSM_FixPh,
    CSM_NUFFT_KspaceMASK_Combined,
    KspaceMask,
    KspaceMask_kz,
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
        KspaceMask_physics=LinearPhysics(),
        max_iter=8,
        tol=2e-2,
        device=torch.device("cuda:0"),
        preconditioning=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.M = MVF_physics
        self.S = CSM_physics
        self.N = NUFFT_physics
        self.MASK = KspaceMask_physics
        self.forward_model = self.MASK * self.N * self.S * self.M

        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.preconditioning = preconditioning
        self.weight = None

    def update_parameters(
        self,
        mvf_kernels=None,
        csm_kernels=None,
        kspace_traj=None,
        kspace_mask=None,
        nufft_im_size=None,
        preconditioner=None,
    ):
        if hasattr(self, "M") and mvf_kernels is not None:
            if isinstance(self.M, Repeat):
                self.M.update_parameters(mvf_kernels.to(self.device))
        if hasattr(self, "S") and csm_kernels is not None:
            if isinstance(self.S, (CSM_FixPh, CSM)):
                self.S.update_parameters(csm_kernels.to(self.device))
        if hasattr(self, "N") and kspace_traj is not None:
            if isinstance(self.N, (NUFFT, NUFFT_2D_FFT_1D, NUFFT_3D)):
                self.N.update_parameters(
                    kspace_traj=kspace_traj.to(self.device),
                    nufft_im_size=nufft_im_size,
                )
            elif isinstance(self.N, CSM_NUFFT_KspaceMASK_Combined):
                # CSM_NUFFT_Combined requires both kspace_traj and csm_kernels
                self.N.update_parameters(
                    kspace_traj=kspace_traj.to(self.device),
                    nufft_im_size=nufft_im_size,
                    csm=csm_kernels.to(self.device),
                    mask=kspace_mask.to(self.device)
                    if kspace_mask is not None
                    else None,
                    preconditioner=preconditioner.to(self.device)
                    if preconditioner is not None
                    else None,
                )
                self.N.preconditioning = self.preconditioning
        if hasattr(self, "MASK") and kspace_mask is not None:
            if isinstance(self.MASK, (KspaceMask_kz, KspaceMask)):
                # Update the mask parameters if it is a LinearPhysics object
                # This is useful for dynamic masks or other mask types
                self.MASK.update_parameters(kspace_mask.to(self.device))
        if self.preconditioning and preconditioner is not None:
            # Update the preconditioner if it is provided
            self.preconditioner = preconditioner.to(self.device)

    def A(self, x):
        input_device = x.device
        x = x.to(self.device)
        y = self.forward_model.A(x)
        return y.to(input_device)

    def A_adjoint(self, y):
        input_device = y.device
        y = y.to(self.device)
        x = self.forward_model.A_adjoint(y)
        return x.to(input_device)

    def A_adjoint_A(self, x, **kwargs):
        input_device = x.device
        if input_device != self.device:
            # Ensure the input is on the correct device
            x = x.to(self.device)
        _x = self.M.A_adjoint(self.N.A_adjoint_A(self.M.A(x), **kwargs))
        if input_device != self.device:
            # Move the result back to the original device
            _x = _x.to(input_device)
        return _x

    def A_dagger(
        self, y, init, solver="CG", max_iter=None, tol=None, verbose=False, **kwargs
    ):
        input_device = y.device
        y = y.to(self.device)
        if max_iter is not None:
            self.max_iter = max_iter
        if tol is not None:
            self.tol = tol
        if solver is not None:
            self.solver = solver
        if init is not None:
            init = init.to(self.device)
        else:
            if self.preconditioning:
                # Use the preconditioned adjoint for initialization
                init = self.A_adjoint(torch.sqrt(self.weight) * y)
            else:
                # Use the standard adjoint for initialization
                init = self.A_adjoint(self.weight * y)
        with torch.no_grad():
            results = least_squares(
                self.A,
                self.A_adjoint,
                y,
                init=init,
                parallel_dim=[0],
                AAT=self.A_A_adjoint,
                verbose=verbose,
                ATA=self.A_adjoint_A,
                max_iter=self.max_iter,
                tol=self.tol,
                solver=self.solver,
                **kwargs,
            )
        init.to(input_device)
        return results.to(input_device)

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

    def sequential_A(self, x):
        input_device = x.device
        x = x.to(self.device)
        image_multi_phase = self.M.A(x)
        image_csm = self.S.A(image_multi_phase)
        y_unmasked = self.N.A(image_csm)
        if hasattr(self, "MASK"):
            y_masked = self.MASK.A(y)
        else:
            y_masked = y_unmasked
        return (
            image_multi_phase.to(input_device),
            image_csm.to(input_device),
            y_unmasked.to(input_device),
            y_masked.to(input_device),
        )

    def sequential_A_adjoint(self, y):
        input_device = y.device
        y = y.to(self.device)
        if hasattr(self, "MASK"):
            y_unmasked = self.MASK.A_adjoint(y)
        else:
            y_unmasked = y
        image_multi_ch = self.N.A_adjoint(y_unmasked)
        image_multi_phase = self.S.A_adjoint(image_multi_ch)
        x = self.M.A_adjoint(image_multi_phase)
        return (
            x.to(input_device),
            image_multi_phase.to(input_device),
            image_multi_ch.to(input_device),
            y_unmasked.to(input_device),
        )


dwunet = ComplexUnet(
    1,
    1,
    spatial_dims=3,
    conv_net=DWUNetSmall(
        in_channels=2,
        out_channels=2,
        features=(16, 32, 64, 128, 256),
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
    if hasattr(physics, "x_init") and physics.x_init is not None:
        x_init = physics.x_init
    else:
        raise ValueError(
            "Physics object must have 'x_init' attribute set to a valid tensor."
        )
    # _x = physics.A_dagger(y, init=x_init)
    _x = x_init
    return {"est": (_x, _x.clone())}
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
        prior = RED(denoiser, unsqueeze_channel_dim=False)
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
        denoiser=ComplexUnet(
            1,
            1,
            spatial_dims=3,
            conv_net=DWUNetSmall(
                in_channels=2,
                out_channels=2,
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
    denoiser,
    denoiser_device=torch.device("cuda:1"),
):
    reconstructor = dinv.models.ArtifactRemoval(denoiser, mode="adjoint")

    def backbone_inference(self, tensor_in, physics, y):
        return self.backbone_net(tensor_in.unsqueeze(1), 1).squeeze(1)

    reconstructor.backbone_inference = MethodType(backbone_inference, reconstructor)
    return reconstructor
