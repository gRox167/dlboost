from typing import Dict

import einx
import tensordict
import torch
from deepinv.physics import Physics
from dlboost.operators.linear_fitting import (
    LinearFittingOperator,
    build_design_matrix,
    build_target_matrix,
)
from torch import Tensor


class MultiFA_T2star_NonLinearOperator(Physics):
    """
    T2* non-linear model for multi–flip-angle data.

    The non-linear model is:
    $
    S_{i,j} = M_i * exp(-TE_j / T2*)
    $
    where $S_{i,j}$ is the signal at flip angle $i$ and echo time $j$.
    """

    def __init__(self, TEs, FAs, device=torch.device("cpu")):
        super().__init__()
        # TEs: 1D tensor [n_echos].
        self.num_echos = len(TEs)
        self.num_flip_angles = len(FAs)
        self.TEs = torch.as_tensor(TEs, dtype=torch.float32)
        self.device = device
        if device is not None:
            self.to(device)

    def update_parameters(self, TEs=None, FAs=None):
        """
        Update the echo times and flip angles.
        """
        if TEs is not None:
            self.TEs = torch.as_tensor(TEs, dtype=torch.float32, device=self.device)
            self.num_echos = len(TEs)
        if FAs is not None:
            self.num_flip_angles = len(FAs)

    # @torch.compile
    def A(self, x: Dict[str, Tensor], **kwargs) -> Tensor:
        """
        x: Dict {
            "T2star": [b, d, h , w],  # T2*
            "M": [b, num_flip_angles, d, h, w], # M_i, the signal at flip angle i
        }
        output shape: [b, num_flip_angles, num_echos, d, h, w]
        """
        if "T2star" in x:
            div = einx.divide("TEs, b d h w -> b TEs d h w", self.TEs, x["T2star"])
        elif "R2star" in x:
            div = einx.dot("TEs, b d h w -> b TEs d h w", self.TEs, x["R2star"])
        else:
            raise ValueError("Input must contain 'T2star' or 'R2star'.")
        S = einx.dot(
            "b FAs ch d h w, b TEs d h w -> b FAs TEs ch d h w",
            x["M"],
            torch.exp(-div),  # type: ignore
        )
        return (
            S.to(torch.complex64)
            if x["M"].dtype == torch.float32 or x["M"].dtype == torch.complex64
            else S.to(torch.complex128)
        )


class MultiFA_T2star_LinearOperator(LinearFittingOperator):
    """
    T2* linear model for multi–flip-angle data.

    The linear model is:
    $
    ln S_{i,j} = ln M_i - TE_j R2*
    $
    where $S_{i,j}$ is the signal at flip angle $i$ and echo time $j$.
    """

    def __init__(self, TEs, FAs, *args, **kwargs):
        super().__init__(
            design_matrix=build_design_matrix(
                flip_angles=FAs,
                echo_times=[-t for t in TEs],
                with_intercept=True,
            ),
            **kwargs,
        )

    def update_parameters(self, TEs=None, FAs=None, **kwargs):
        """
        Update the echo times and flip angles.
        """
        if TEs is not None and FAs is not None:
            super().update_parameters(
                design_matrix=build_design_matrix(
                    flip_angles=FAs,
                    echo_times=[-t for t in TEs],
                    with_intercept=True,
                    device=self.device,
                ),
                **kwargs,
            )
        # update kwargs
        if "mask" in kwargs and kwargs["mask"] is not None:
            self.mask = kwargs["mask"]

    # @torch.compile
    def A(self, x: Dict[str, Tensor], **kwargs) -> Tensor:
        """
        x: Dict {
            "R2star": [b, d, h , w],  # R2*
            "M": [b, num_flip_angles, d, h, w], # M_i, the signal at flip angle i
        }
        output shape: [b, num_flip_angles, num_echos, d, h, w]
        """
        if "T2star" in x:
            raise ValueError(
                "This operator is linear, so it expects 'R2star' instead of 'T2star'."
            )
        elif "R2star" in x:
            stacked_x = einx.rearrange(
                "b d h w, b FAs d h w -> b (1+FAs) d h w",
                x["R2star"],
                x["M"],
            )
            return super().A(stacked_x)

    def A_adjoint(self, S: Tensor, **kwargs) -> Dict[str, Tensor]:
        """
        Adjoint of the linear operator.
        """
        y = build_target_matrix(S, self.device)
        stacked_x = super().A_adjoint(y)
        return {
            "R2star": stacked_x[:, 0],
            "M": stacked_x[:, 1:],
        }

    def A_dagger(self, y, mask=None, **kwargs):
        y = build_target_matrix(y, self.device)
        stacked_x = super().A_dagger(y, mask=mask, **kwargs)
        return {
            "R2star": stacked_x[:, 0],
            "M": stacked_x[:, 1:],
        }


class MultiFA_T2star_Field_Correction_NonLinearOperator(
    MultiFA_T2star_NonLinearOperator
):
    """
    T2* non-linear model for multi–flip-angle data with field correction.
    The non-linear model is:
    $
    S_{i,j} = M_i * exp(-TE_j / T2*) * exp(i * (phi_0 + TE_j * \Delta f))
    $
    where $S_{i,j}$ is the signal at flip angle $i$ and echo time $j$,
    $\Delta f$ is the field offset, and $\phi_0$ is the initial
    phase offset.
    """

    def __init__(self, TEs, FAs, device=torch.device("cpu"), phase_init=False):
        super().__init__(TEs, FAs, device)
        self.phase_init = phase_init

    def update_parameters(self, TEs=None, FAs=None):
        """
        Update the echo times, flip angles, and initial phase offset.
        """
        super().update_parameters(TEs=TEs, FAs=FAs)

    def A(self, x: Dict[str, Tensor], **kwargs) -> Tensor:
        """
        x: Dict {
            "T2star": [b, d, h , w],  # T2*
            "M": [b, num_flip_angles, d, h, w], # M_i, the spin density at flip angle i, real number
            "phi_0": [b, d, h, w],  # Initial phase offset, same for all flip angles
            "Delta_f": [b, d, h, w],  # Field offset, same for all flip angles
        }
        output shape: [b, num_flip_angles, num_echos, d, h, w]
        """
        if "T2star" in x:
            div = -einx.divide("TEs, b d h w -> b TEs d h w", self.TEs, x["T2star"])
        elif "R2star" in x:
            div = -einx.dot(
                "TEs, b d h w -> b TEs d h w",
                self.TEs,
                x["R2star"],
            )
        else:
            raise ValueError("Input must contain 'T2star' or 'R2star'.")
        if self.phase_init:
            phase_correction = 1j * (
                x["phi_0"][:, None]
                + einx.dot("TEs, b d h w -> b TEs d h w", self.TEs, x["Delta_f"])
            )
        else:
            phase_correction = 1j * einx.dot(
                "TEs, b d h w -> b TEs d h w", self.TEs, x["Delta_f"]
            )
        exp_term = torch.exp(div + phase_correction)
        S = einx.dot(
            "b FAs d h w, b TEs d h w -> b FAs TEs d h w",
            x["M"],
            exp_term,
        )
        return S


class MultiFA_T2star_Field_Correction_LinearOperator(MultiFA_T2star_NonLinearOperator):
    r"""
    T2* non-linear model for multi–flip-angle data with field correction.
    The non-linear model is:
    $
    S_{i,j} = M_i * exp(-TE_j / T2*) * exp(i * (phi_0 + TE_j * \Delta f))
    $
    where $S_{i,j}$ is the signal at flip angle $i$ and echo time $j$,
    $\Delta f$ is the field offset, and $\phi_0$ is the initial
    phase offset.
    """

    def __init__(self, TEs, FAs, device=torch.device("cpu"), phase_init=False):
        super().__init__(TEs, FAs, device)
        self.phase_init = phase_init

    def update_parameters(self, TEs=None, FAs=None):
        """
        Update the echo times, flip angles, and initial phase offset.
        """
        super().update_parameters(TEs=TEs, FAs=FAs)

    def A(self, x: Dict[str, Tensor], **kwargs) -> Tensor:
        """
        x: Dict {
            "T2star": [b, d, h , w],  # T2*
            "M": [b, num_flip_angles, d, h, w], # M_i, the spin density at flip angle i, real number
            "phi_0": [b, d, h, w],  # Initial phase offset, same for all flip angles
            "Delta_f": [b, d, h, w],  # Field offset, same for all flip angles
        }
        output shape: [b, num_flip_angles, num_echos, d, h, w]
        """
        if "T2star" in x:
            raise ValueError(
                "This operator is linear, so it expects 'Log_R2star' instead of 'T2star'."
            )
        elif "Log_R2star" in x:
            div = einx.elementwise(
                "b d h w, TEs -> b TEs d h w",
                x["Log_R2star"],
                -self.TEs,
                op=torch.pow,
            )
        else:
            raise ValueError("Input must contain 'Log_R2star'.")
        if self.phase_init:
            phase_correction = 1j * (
                x["phi_0"][:, None]
                + einx.dot("TEs, b d h w -> b TEs d h w", self.TEs, x["Delta_f"])
            )
        else:
            phase_correction = 1j * einx.dot(
                "TEs, b d h w -> b TEs d h w", self.TEs, x["Delta_f"]
            )
        exp_term = div * torch.exp(phase_correction)
        S = einx.dot(
            "b FAs d h w, b TEs d h w -> b FAs TEs d h w",
            x["M"],
            exp_term,
        )
        return S


def sigmoid(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Beta scaled sigmoid function."""
    return torch.nn.functional.sigmoid(beta * x)


def sigmoid_inverse(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Inverse of `sigmoid`."""
    return torch.log(x / (1 - x)) / beta


def softplus(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Beta scaled softplus function."""
    return -(1 / beta) * torch.nn.functional.logsigmoid(-beta * x)


def softplus_inverse(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Inverse of `softplus`."""
    return x + torch.log(-torch.expm1(-beta * x)) / beta


class Constrain(Physics):
    """Transformation to map real-valued tensors to certain ranges."""

    def __init__(
        self,
        bounds: Dict[str, tuple[float | None, float | None]],
        beta_sigmoid: float = 1.0,
        beta_softplus: float = 1.0,
    ) -> None:
        """Initialize a constraint operator.

        The operator maps real-valued tensors to certain ranges. The transformation is applied element-wise.
        The transformation is defined by the bounds. The bounds are applied in the order of the input tensors.
        If there are more input tensors than bounds, the remaining tensors are passed through without transformation.

        If an input tensor is bounded from below AND above, a sigmoid transformation is applied.
        If an input tensor is bounded from below OR above, a softplus transformation is applied.

        If an input is complex valued, the bounds are to the real and imaginary parts separately,
        i.e., for bounds (a,b), the complex number is constrained to a rectangle in the complex plane
        with corners a+ai, a+bi, b+ai, b+bi.

        Parameters
        ----------
        bounds
            Sequence of (lower_bound, upper_bound) values. If a bound is None, the value is not constrained.
            If a lower bound is -inf, the value is not constrained from below. If an upper bound is inf,
            the value is not constrained from above.
            If the bounds are set to (None, None) or (-inf, inf), the value is not constrained at all.
        beta_sigmoid
            beta parameter for the sigmoid transformation (used if an input has two bounds).
            A higher value leads to a steeper sigmoid.
        beta_softplus
            parameter for the softplus transformation (used if an input is either bounded from below or above).
            A higher value leads to a steeper softplus.

        Raises
        ------
        ValueError
            If the lower bound is greater than the upper bound.
        ValueError
            If the a bound is nan.
        ValueError
            If the parameter beta_sigmoid and beta_softplus are not greater than zero.
        """
        super().__init__()

        if beta_sigmoid <= 0:
            raise ValueError(
                f"parameter beta_sigmoid must be greater than zero; given {beta_sigmoid}"
            )
        if beta_softplus <= 0:
            raise ValueError(
                f"parameter beta_softplus must be greater than zero; given {beta_softplus}"
            )

        self.beta_sigmoid = beta_sigmoid
        self.beta_softplus = beta_softplus
        self.lower_bounds = {
            k: torch.as_tensor(-torch.inf if lb is None else lb)
            for k, (lb, ub) in bounds.items()
        }
        self.upper_bounds = {
            k: torch.as_tensor(torch.inf if ub is None else ub)
            for k, (lb, ub) in bounds.items()
        }

        # for lb, ub in zip(self.lower_bounds, self.upper_bounds, strict=True):
        #     if lb.isnan():
        #         raise ValueError("nan is invalid as lower bound.")
        #     if ub.isnan():
        #         raise ValueError("nan is invalid as upper bound.")
        #     if lb >= ub:
        #         raise ValueError(
        #             "bounds should be ( (a1,b1), (a2,b2), ...) with ai < bi if neither ai or bi is None;"
        #             f"\nbound tuple {lb, ub} is invalid as the lower bound is higher than the upper bound",
        #         )

    def _apply_forward(
        self, item: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor
    ) -> torch.Tensor:
        """Apply the forward transformation to the input tensor."""
        if item.dtype.is_complex:
            real = self._apply_forward(item.real, lb, ub)
            imag = self._apply_forward(item.imag, lb, ub)
            return torch.complex(real, imag)

        if not lb.isneginf() and not ub.isposinf():
            # print(lb + (ub - lb) * sigmoid(item, beta=self.beta_sigmoid))
            return lb + (ub - lb) * sigmoid(item, beta=self.beta_sigmoid)

        if not lb.isneginf():
            # bounds are (lb,inf)
            return lb + softplus(item, beta=self.beta_softplus)

        if not ub.isposinf():
            # bounds are (-inf,ub)
            return ub - softplus(-item, beta=self.beta_softplus)

        return item  # unconstrained case

    def A(self, x: tensordict.TensorDict) -> tensordict.TensorDict:
        """Transform tensors to chosen range.

        Parameters
        ----------
        x
            tensordict to be transformed

        Returns
        -------
            tensors transformed to the range defined by the chosen bounds
        """
        d = {}
        for k, v in x.items():
            d[k] = self._apply_forward(v, self.lower_bounds[k], self.upper_bounds[k])
        return tensordict.TensorDict(d)

    def _apply_inverse(
        self, item: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor
    ) -> torch.Tensor:
        if item.dtype.is_complex:
            real = self._apply_inverse(item.real, lb, ub)
            imag = self._apply_inverse(item.imag, lb, ub)
            return torch.complex(real, imag)

        if not lb.isneginf() and not ub.isposinf():
            # bounds are (lb,ub)
            print(sigmoid_inverse((item - lb) / (ub - lb)))
            return sigmoid_inverse((item - lb) / (ub - lb), beta=self.beta_sigmoid)

        if not lb.isneginf():
            # bounds are (lb,inf)
            return softplus_inverse(item - lb, beta=self.beta_softplus)

        if not ub.isposinf():
            # bounds are (-inf,ub)
            return -softplus_inverse(-(item - ub), beta=self.beta_softplus)

        return item  # unconstrained case

    def A_inv(self, x: tensordict.TensorDict) -> tensordict.TensorDict:
        """Inverse transformation of tensors to chosen range.

        Parameters
        ----------
        x
            tensordict to be transformed

        Returns
        -------
            tensors transformed to the unconstrained range
        """
        d = {}
        for k, v in x.items():
            d[k] = self._apply_inverse(v, self.lower_bounds[k], self.upper_bounds[k])
        return tensordict.TensorDict(d)


# class MultiFA_T2star_LinearOperator(LinearPhysics):
#     """
#     T2* linear model for multi–flip-angle data.

#     The linear model is:
#     $
#     log(S_{i,j}) = log(M_i) - TE_j * R2*
#     $

#     where $S_{i,j}$ is the signal at flip angle $i$ and echo time $j$.
#     """

#     def __init__(self, TEs, FAs):
#         super().__init__()
#         # TEs: 1D tensor [n_echos].
#         self.num_echos = len(TEs)
#         self.num_flip_angles = len(FAs)
#         self.register_buffer("TEs", TEs[None, :, None, None, None])

#     def forward(self, x: tensordict.TensorDict) -> Tensor:
#         """
#         x: Dict {
#             "R2star": [b, d, h , w],  # R2*
#             "M_log": [b, num_flip_angles, d, h, w], # M_i, the signal at flip angle i
#         }
#         output shape: [b, num_flip_angles, num_echos, d, h, w]
#         """
#         M_log = x["M_log"]
#         # Broadcasting T2star to match M's shape
#         S_log = M_log - self.TEs * x["R2star"]
#         return S_log

#     def adjoint(self, x):
