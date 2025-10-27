from typing import Dict, Iterable, Optional

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


class Init_Phase_Correction(Physics):
    """
    Initial phase correction operator that applies constant phase offset to input signals.

    The phase correction model is:
    $
    S_{corrected} = S_{input} \cdot exp(i \cdot \phi_0)
    $
    where $S_{input}$ is the input signal and $\phi_0$ is the initial phase offset.

    Example usage:
    ```python
    # Basic usage
    init_phase_op = Init_Phase_Correction(
        device=torch.device("cuda:0")
    )

    # With per-coil phase initialization
    init_phase_op = Init_Phase_Correction(
        phase_init_for_each_channel=True,
        global_phase_init=False
    )

    # Apply initial phase correction to signal
    input_signal = torch.randn(1, 3, 4, 288, 288, dtype=torch.complex64)  # [b, FAs, TEs, d, h, w]
    phi_0 = torch.randn(1, 8, 288, 288)  # [b, ch, d, h, w] for 8 coils
    corrected_signal = init_phase_op.A({"signal": input_signal, "phi_0": phi_0})
    ```
    """

    def __init__(
        self,
        device=torch.device("cpu"),
        phase_init_for_each_FA=False,
        phase_init_for_each_channel=False,
        global_phase_init=True,
    ):
        super().__init__()
        self.phase_init_for_each_FA = phase_init_for_each_FA
        self.phase_init_for_each_channel = phase_init_for_each_channel
        self.global_phase_init = global_phase_init
        self.device = device

    def A(self, x: Dict[str, Tensor], **kwargs) -> Tensor:
        """
        Apply initial phase correction to input signal.

        Args:
            x: Dict containing:
                "signal": Input signal tensor from T2* model
                    Shape: [b, FAs, TEs, d, h, w]
                "phi_0": Initial phase offset
                    - [b, d, h, w] if not phase_init_for_each_FA and not phase_init_for_each_channel
                    - [b, FAs, d, h, w] if phase_init_for_each_FA
                    - [b, ch, d, h, w] if phase_init_for_each_channel

        Returns:
            corrected_signal: Phase-corrected signal
                Shape: [b, FAs, TEs, ch, d, h, w] when phase_init_for_each_channel=True
                Shape: [b, FAs, TEs, (), d, h, w] otherwise
        """
        signal_input = x["signal"]

        if "phi_0" not in x:
            # If no phase correction provided, just expand channel dimension if needed
            if self.phase_init_for_each_channel:
                return einx.rearrange(
                    "b FAs TEs d h w -> b FAs TEs () d h w", signal_input
                )
            else:
                return signal_input

        # phi_0 = x["phi_0"].expand_as(signal_input[..., 0, :, :, :])

        # Create phase correction factor
        phase_factor = torch.exp(1j * x["phi_0"])
        d, h, w = signal_input.shape[-3:]

        # Handle reshaping and application based on configuration
        if self.phase_init_for_each_channel:
            return einx.dot(
                "b FAs TEs d h w, b ch d h w -> b FAs TEs ch d h w",
                signal_input,
                phase_factor.expand(-1, -1, d, h, w),
            )
        elif self.phase_init_for_each_FA:
            return einx.dot(
                "b FAs TEs d h w, b FAs d h w -> b FAs TEs () d h w",
                signal_input,
                phase_factor.expand(-1, -1, d, h, w),
            )
        else:
            return einx.dot(
                "b FAs TEs d h w, b d h w -> b FAs TEs () d h w",
                signal_input,
                phase_factor.expand(-1, d, h, w),
            )


class Field_Offset_Phase_Correction(Physics):
    """
    Field offset phase correction operator that applies frequency-dependent phase offset.

    The phase correction model is:
    $
    \phi = TE \cdot \Delta f
    $
    where $\Delta f$ is the field offset and $TE$ are the echo times.

    Example usage:
    ```python
    # Basic usage
    field_phase_op = Field_Offset_Phase_Correction(
        TEs=[0.002, 0.004, 0.006],  # Echo times in seconds
        device=torch.device("cuda:0")
    )

    # With per-FA field correction
    field_phase_op = Field_Offset_Phase_Correction(
        TEs=[0.002, 0.004, 0.006],
        Delta_f_for_each_FA=True
    )

    # Apply field offset phase correction
    params = {"Delta_f": torch.randn(1, 288, 288)}  # [b, d, h, w]
    phase_factor = field_phase_op.A(params)  # Returns complex exponential
    ```
    """

    def __init__(
        self,
        TEs,
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.TEs = torch.tensor(TEs, dtype=torch.float32, device=device)
        self.num_echos = len(TEs)
        self.device = device

    def update_parameters(self, TEs=None):
        """Update the echo times."""
        if TEs is not None:
            self.TEs = torch.tensor(TEs, dtype=torch.float32, device=self.device)
            self.num_echos = len(TEs)

    def A(self, x: Dict[str, Tensor], **kwargs) -> Tensor:
        """
        Apply field offset phase correction.

        Args:
            x: Dict containing:
                "Delta_f": Field offset
                    - [b, d, h, w] if not Delta_f_for_each_FA
                    - [b, FAs, d, h, w] if Delta_f_for_each_FA

        Returns:
            phase_correction: Complex phase correction factor exp(i * TE * Delta_f)
                Shape: [b, FAs, TEs, d, h, w] or [b, (), TEs, d, h, w]
                Note: No channel dimension for field offset (tissue property, not coil-dependent)
        """
        # Handle Delta_f reshaping and compute frequency term
        if x["Delta_f"].ndim == 4:
            Delta_f = einx.rearrange("b d h w -> b () () d h w", x["Delta_f"])
            freq_term = einx.dot(
                "TEs, b () () d h w -> b () TEs d h w", self.TEs, Delta_f
            )
        elif x["Delta_f"].ndim == 5:
            Delta_f = einx.rearrange("b FAs d h w -> b FAs () d h w", x["Delta_f"])
            freq_term = einx.dot(
                "TEs, b FAs () d h w -> b FAs TEs d h w", self.TEs, Delta_f
            )

        return torch.exp(1j * 2 * torch.pi * freq_term)


class LinearGradientIntravoxelDephasing(Physics):
    """
    Linear-gradient intravoxel dephasing operator using the product-sinc model.

    C_e = prod_d sinc(pi * TE * g_d * L_d)

    where g_d are the components of the spatial gradient of Delta_f (Hz/mm) and
    L_d are the physical voxel dimensions (mm). The operator returns the magnitude
    attenuation factor per echo.
    """

    _GRADIENT_KEYS = ("Delta_f_grad", "Delta_f_gradient", "grad_Delta_f")

    def __init__(
        self,
        TEs: Iterable[float],
        voxel_dimensions_mm: Iterable[float] | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.echo_times = torch.tensor(list(TEs), dtype=torch.float32, device=device)
        self.num_echo_times = len(self.echo_times)

        if voxel_dimensions_mm is None:
            voxel_dimensions_mm = (1.0, 1.0, 1.0)
        voxel_tensor = torch.tensor(
            list(voxel_dimensions_mm), dtype=torch.float32, device=device
        )
        if voxel_tensor.numel() != 3:
            raise ValueError("voxel_dimensions_mm must provide exactly three values.")
        self.voxel_dimensions = voxel_tensor

    def update_parameters(
        self,
        TEs: Optional[Iterable[float]] = None,
        voxel_dimensions_mm: Optional[Iterable[float]] = None,
    ):
        if TEs is not None:
            self.echo_times = torch.tensor(
                list(TEs), dtype=torch.float32, device=self.device
            )
            self.num_echo_times = len(self.echo_times)
        if voxel_dimensions_mm is not None:
            voxel_tensor = torch.tensor(
                list(voxel_dimensions_mm), dtype=torch.float32, device=self.device
            )
            if voxel_tensor.numel() != 3:
                raise ValueError(
                    "voxel_dimensions_mm must provide exactly three values."
                )
            self.voxel_dimensions = voxel_tensor

    @staticmethod
    def _select_gradient(x: Dict[str, Tensor]) -> Tensor:
        for key in LinearGradientIntravoxelDephasing._GRADIENT_KEYS:
            if key in x:
                return x[key]
        raise ValueError(
            "LinearGradientIntravoxelDephasing requires a gradient field with one of "
            f"keys {LinearGradientIntravoxelDephasing._GRADIENT_KEYS}."
        )

    def A(self, x: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor]:
        """
        Args:
            x: Dict containing the spatial gradient of Delta_f with shape [b, 3, d, h, w]
               under any recognised gradient key.

        Returns:
            dict {"C_e": Tensor} with shape [b, num_echo_times, d, h, w] containing the
            magnitude attenuation factors per echo.
        """
        grad = self._select_gradient(x).to(self.device)
        if grad.ndim != 5 or grad.shape[1] != 3:
            raise ValueError(
                "Gradient tensor must have shape [batch, 3, depth, height, width]."
            )

        # Scale gradient by voxel dimensions (Hz/mm * mm -> Hz)
        scaled_grad = grad * self.voxel_dimensions.view(1, 3, 1, 1, 1)

        # Broadcast echo times to match spatial dimensions: [1, 1, TEs, 1, 1, 1]
        te = self.echo_times.view(1, 1, -1, 1, 1, 1)
        argument = torch.pi * scaled_grad.unsqueeze(2) * te  # [b, 3, TEs, d, h, w]

        sinc_terms = torch.sinc(argument / torch.pi)
        attenuation = torch.prod(sinc_terms, dim=1).abs()  # [b, TEs, d, h, w]
        return {"C_e": attenuation}


class MultiFA_T2star_NonLinearOperator(Physics):
    """
    T2* non-linear model for multi–flip-angle data.

    The non-linear model is:
    $
    S_{i,j} = M_i * exp(-TE_j / T2*)
    $
    where $S_{i,j}$ is the signal at flip angle $i$ and echo time $j$.
    """

    def __init__(self, FAs, TEs, device=torch.device("cpu")):
        super().__init__()
        # TEs: 1D tensor [n_echos].
        self.num_flip_angles = len(FAs)
        self.num_echos = len(TEs)
        self.TEs = torch.as_tensor(TEs, dtype=torch.float32, device=device)
        self.device = device
        if device is not None:
            self.to(device)

    def update_parameters(self, FAs=None, TEs=None):
        """
        Update the echo times and flip angles.
        """
        if TEs is not None:
            self.TEs = torch.tensor(TEs, dtype=torch.float32, device=self.device)
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
        # S = einx.dot(
        #     "b FAs ch d h w, b TEs d h w -> b FAs TEs ch d h w",
        #     x["M"],
        #     torch.exp(-div),  # type: ignore
        # )
        S = einx.dot(
            "b FAs d h w, b TEs d h w -> b FAs TEs d h w",
            x["M"],
            torch.exp(-div),  # type: ignore
        )
        # return S
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

    def __init__(self, FAs, TEs, *args, **kwargs):
        super().__init__(
            design_matrix=build_design_matrix(
                flip_angles=FAs,
                echo_times=[-t for t in TEs],
                with_intercept=True,
            ),
            **kwargs,
        )

    def update_parameters(self, FAs=None, TEs=None, **kwargs):
        """
        Update the echo times and flip angles.
        """
        if FAs is not None and TEs is not None:
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
    def A(self, x: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor] | None:
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
    S_{m,n} = M_m * exp(-TE_n / T2*) * exp(i * TE_n * \Delta f)
    $
    where $S_{i,j}$ is the signal at flip angle $i$ and echo time $j$,
    and $\Delta f$ is the field offset.

    Note: Initial phase correction (phi_0) should be handled by a separate
    Init_Phase_Correction operator for per-coil flexibility.
    """

    def __init__(
        self,
        FAs,
        TEs,
        device=torch.device("cpu"),
        # Delta_f_for_each_FA=False,
        field_offset_phase_op=Field_Offset_Phase_Correction,
    ):
        """
        Initialize the field correction non-linear operator.

        Args:
            FAs: List of flip angles
            TEs: List of echo times
            device: Device for tensor allocation ('cpu' or 'cuda')
            # Delta_f_for_each_FA: Whether to use separate field offset for each flip angle
        """
        super().__init__(FAs, TEs, device)
        # self.Delta_f_for_each_FA = Delta_f_for_each_FA

        # Initialize field offset phase correction operator
        if field_offset_phase_op is not None:
            self.field_offset_phase_op = field_offset_phase_op(
                TEs=TEs,
                device=device,
                # Delta_f_for_each_FA=Delta_f_for_each_FA,
            )

    def update_parameters(self, FAs=None, TEs=None):
        """
        Update the echo times and flip angles.
        """
        super().update_parameters(FAs=FAs, TEs=TEs)
        if hasattr(self, "field_offset_phase_op"):
            self.field_offset_phase_op.update_parameters(TEs=TEs)

    def A(self, x: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor | None]:
        """
        x: Dict {
            "T2star": [b, d, h , w],  # T2*
            "M": [b, num_flip_angles, d, h, w], # M_i, the spin density at flip angle i
            "Delta_f": [b, d, h, w],  # Field offset
        }
        output shape: [b, num_flip_angles, num_echos, d, h, w]
        """
        # Compute T2* decay term (div has shape [b, 1, TEs, d, h, w])
        if "T2star" in x:
            div = -einx.divide("TEs, b d h w -> b () TEs d h w", self.TEs, x["T2star"])
        elif "R2star" in x:
            div = -einx.dot(
                "TEs, b d h w -> b () TEs d h w",
                self.TEs,
                x["R2star"],
            )
        else:
            raise ValueError("Input must contain 'T2star' or 'R2star'.")

        # Accept either pre-computed ME1 or raw M (per-FA proton density)
        if "ME1" in x:
            ME1 = einx.rearrange(
                "b FAs d h w -> b FAs () d h w", x["ME1"]
            )  # precomputed M*E1
            magnitude_decay = ME1 * torch.exp(div)
        elif "M" in x:
            M = einx.rearrange("b FAs d h w -> b FAs () d h w", x["M"])  # M per FA
            magnitude_decay = M * torch.exp(div)
        else:
            raise ValueError("Input must contain 'M' or 'ME1' (pre-computed M*E1).")

        # Apply field offset correction if available
        if hasattr(self, "field_offset_phase_op"):
            field_phase_factor = self.field_offset_phase_op.A(x)
            S = magnitude_decay * field_phase_factor
        else:
            S = magnitude_decay

        return {
            "signal": S,
            "phi_0": x.get("phi_0", None),  # Pass through initial phase if present
        }


class T1_NonLinearOperator(Physics):
    """
    T1 operator that returns the Ernst factor E1 per flip-angle only.

    E1(R1, alpha) = sin(alpha) * (1 - exp(-TR * R1)) / (1 - cos(alpha) * exp(-TR * R1))

    Inputs:
      - x["R1"]  : [b, d, h, w]  (preferred) OR
      - x["T1"]  : [b, d, h, w]  (will be inverted to R1)

    Returns:
      - dict {"E1": Tensor} where E1 shape is [b, FAs, d, h, w] (float)
    """

    def __init__(
        self,
        flip_angles: Iterable[float],
        TR: float,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.flip_angles = torch.tensor(
            list(flip_angles), dtype=torch.float32, device=device
        )
        self.alphas_rad = self.flip_angles * torch.pi / 180.0
        self.num_flip_angles = len(self.flip_angles)
        self.TR = TR
        self.device = device
        if device is not None:
            self.to(device)

    def update_parameters(
        self,
        flip_angles: Optional[Iterable[float]] = None,
        TR: Optional[float] = None,
        b1_map: Optional[Tensor] = None,
    ):
        if flip_angles is not None:
            self.flip_angles = torch.tensor(
                list(flip_angles), dtype=torch.float32, device=self.device
            )
            self.alphas_rad = self.flip_angles * torch.pi / 180.0
            self.num_flip_angles = len(self.flip_angles)
        if TR is not None:
            self.TR = TR
        if b1_map is not None:
            self.b1_map = b1_map.to(self.device)
            self.alphas_rad = einx.dot(
                "FAs, b d h w -> b FAs d h w", self.alphas_rad, self.b1_map
            )

    def A(self, x: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor]:
        if "R1" in x:
            R1 = x["R1"]
        elif "T1" in x:
            T1 = torch.clamp(x["T1"], min=1e-6)
            R1 = 1.0 / T1
        else:
            raise ValueError(
                "T1_NonLinearOperator requires 'R1' or 'T1' in input dict."
            )

        # compute exp(-TR * R1) shape [b, d, h, w]
        expTRR1 = torch.exp(-self.TR * R1)

        # prepare sin and cos arrays for broadcasting
        sin_alphas = torch.sin(self.alphas_rad).to(self.device)  # [FAs]
        cos_alphas = torch.cos(self.alphas_rad).to(self.device)  # [FAs]

        # broadcast expTRR1 to [b, () , () , d, h, w] for einx.dot usage
        exp_b = einx.rearrange("b d h w -> b () () d h w", expTRR1)
        if self.b1_map is not None:
            numer = einx.dot(
                "b FAs d h w, b () () d h w -> b FAs () d h w",
                sin_alphas,
                (1.0 - exp_b),
            )
            denom = 1.0 - einx.dot(
                "b FAs d h w, b () () d h w -> b FAs () d h w", cos_alphas, exp_b
            )
        else:
            numer = einx.dot(
                "FAs, b () () d h w -> b FAs () d h w", sin_alphas, (1.0 - exp_b)
            )
            denom = 1.0 - einx.dot(
                "FAs, b () () d h w -> b FAs () d h w", cos_alphas, exp_b
            )
        denom = torch.clamp(denom, min=1e-12)

        E1 = numer / denom  # [b, FAs, () , d, h, w]
        E1 = einx.rearrange("b FAs () d h w -> b FAs d h w", E1)
        return {"E1": E1}


class ProtonDensityOperator(Physics):
    """
    Proton density operator that multiplies a scalar M to the input signal.
    S = M * input_signal

    The operator handles broadcasting M to the shape of the input signal.
    For example:
    - M [b, d, h, w], signal [b, FAs, d, h, w] -> M is unsqueezed to [b, 1, d, h, w]
    - M [b, FAs, d, h, w], signal [b, FAs, TEs, d, h, w] -> M is unsqueezed to [b, FAs, 1, d, h, w]

    Inputs:
      - x["M"]  : Proton density map, e.g., [b, d, h, w] or [b, FAs, d, h, w]
      - x[signal_key]: Input signal, e.g., [b, FAs, d, h, w] or [b, FAs, TEs, d, h, w]

    Returns:
      - dict {"S": Tensor} with the same shape as the input signal.
    """

    def __init__(
        self,
        signal_key: str = "E1",
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.signal_key = signal_key
        self.device = device
        if device is not None:
            self.to(device)

    def A(self, x: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor]:
        if "M" not in x:
            raise ValueError("Proton_density_operator requires 'M' in input dict.")
        if self.signal_key not in x:
            raise ValueError(
                f"Proton_density_operator requires '{self.signal_key}' in input dict."
            )

        M = x["M"]
        input_signal = x[self.signal_key]

        # Determine the pattern for einx rearrange
        m_rank = M.ndim
        s_rank = input_signal.ndim

        if m_rank >= s_rank:
            M_expanded = M
        else:
            # Add singleton dimensions to M to match the rank of the signal
            # Assumes the batch dimension is first and spatial dimensions are last.
            num_new_dims = s_rank - m_rank
            # e.g., 'b d h w -> b 1 1 d h w' for 2 new dims
            pattern = f"b ... d h w -> b {' '.join(['()' for _ in range(num_new_dims)])} ... d h w"
            M_expanded = einx.rearrange(pattern, M)

        S = M_expanded * input_signal
        return {"S": S}


def sigmoid(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Beta scaled sigmoid function."""
    return torch.nn.functional.sigmoid(beta * x)


def sigmoid_inverse(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Inverse of `sigmoid`."""
    return torch.log(x / (1 - x)) / beta


def softplus(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Beta scaled softplus function."""
    # return -(1 / beta) * torch.nn.functional.logsigmoid(-beta * x)
    return -torch.nn.functional.logsigmoid(-beta * x)


def softplus_inverse(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Inverse of `softplus`."""
    return x + torch.log(-torch.expm1(-x)) / beta


class Constrain(Physics):
    """Transformation to map real-valued tensors to certain ranges."""

    def __init__(
        self,
        bounds: Dict[str, tuple[float | None, float | None]],
        beta: Dict[str, float] | float = 1.0,
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
        beta
            beta parameter for the transformation (used for both sigmoid and softplus).
            Can be a single float value applied to all parameters, or a dictionary mapping
            parameter names to specific beta values. A higher value leads to a steeper transformation.

        Raises
        ------
        ValueError
            If the lower bound is greater than the upper bound.
        ValueError
            If the a bound is nan.
        ValueError
            If the parameter beta is not greater than zero.
        """
        super().__init__()

        # Handle beta as either dict or float
        if isinstance(beta, dict):
            self.beta = {}
            for k in bounds.keys():
                v = beta.get(k, 1.0)
                if v <= 0:
                    raise ValueError(
                        f"parameter beta[{k}] must be greater than zero; given {v}"
                    )
                self.beta[k] = v
        elif isinstance(beta, (float, int)):
            if beta <= 0:
                raise ValueError(
                    f"parameter beta must be greater than zero; given {beta}"
                )
            self.beta = beta
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
        self, item: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor, param_name: str
    ) -> torch.Tensor | None:
        """Apply the forward transformation to the input tensor."""
        # Get beta value for this parameter
        beta_val = self.beta[param_name] if isinstance(self.beta, dict) else self.beta

        if item.dtype.is_complex:
            real = self._apply_forward(item.real, lb, ub, param_name)
            imag = self._apply_forward(item.imag, lb, ub, param_name)
            return torch.complex(real, imag)

        if not lb.isneginf() and not ub.isposinf():
            # print(lb + (ub - lb) * sigmoid(item, beta=beta_sig))
            return lb + (ub - lb) * sigmoid(item, beta=beta_val)

        if not lb.isneginf():
            # bounds are (lb,inf)
            return lb + softplus(item, beta=beta_val)

        if not ub.isposinf():
            # bounds are (-inf,ub)
            return ub - softplus(-item, beta=beta_val)

        if lb.isneginf() and ub.isposinf():
            # unconstrained case
            return item * beta_val

        # return item  # unconstrained case

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
            d[k] = self._apply_forward(v, self.lower_bounds[k], self.upper_bounds[k], k)
        return tensordict.TensorDict(d)

    def _apply_inverse(
        self, item: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor, param_name: str
    ) -> torch.Tensor | None:
        # Get beta value for this parameter
        beta_val = self.beta[param_name] if isinstance(self.beta, dict) else self.beta

        if item.dtype.is_complex:
            real = self._apply_inverse(item.real, lb, ub, param_name)
            imag = self._apply_inverse(item.imag, lb, ub, param_name)
            return torch.complex(real, imag)

        if not lb.isneginf() and not ub.isposinf():
            # bounds are (lb,ub)
            return sigmoid_inverse((item - lb) / (ub - lb), beta=beta_val)

        if not lb.isneginf():
            # bounds are (lb,inf)
            return softplus_inverse(item - lb, beta=beta_val)

        if not ub.isposinf():
            # bounds are (-inf,ub)
            return -softplus_inverse(-(item - ub), beta=beta_val)

        if lb.isneginf() and ub.isposinf():
            # unconstrained case
            return item / beta_val

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
            d[k] = self._apply_inverse(v, self.lower_bounds[k], self.upper_bounds[k], k)
        return tensordict.TensorDict(d)


class MRI_Signal_Model_Operator(Physics):
    """
    Composite MRI signal model operator that implements the full signal equation:

    s_{a,e,c}(θ) = M·E_1(R_1)·C_e·exp(-TE_e R_2^*)·exp(j(φ_c + 2π TE_e Δf))

    where:
    - E_1(R_1) = sin(α)·(1 - exp(-TR·R_1)) / (1 - cos(α)·exp(-TR·R_1))
    - Parameters: θ=(R_1, R_2^*, Δf, M, φ_c)
    - Sequence parameters: TE_e, TR, α

    This operator combines:
    1. T1_NonLinearOperator for Ernst angle factor computation
    2. Proton density scaling
    3. MultiFA_T2star_Field_Correction_NonLinearOperator for T2* decay and field correction
    4. LinearGradientIntravoxelDephasing for intravoxel static dephasing attenuation
    5. Coil phase correction

    Inputs:
        x: Dict containing:
            "R1" or "T1": [b, d, h, w] - Longitudinal relaxation rate/time
            "R2star" or "T2star": [b, d, h, w] - Transverse relaxation rate/time
            "Delta_f": [b, d, h, w] - Field offset in Hz
            "Delta_f_grad": [b, 3, d, h, w] - Field gradient in Hz/mm (optional, aliases: "Delta_f_gradient", "grad_Delta_f"; computed from "Delta_f" when omitted)
            "M": [b, d, h, w] - Proton density map
            "phi_c": [b, coils, d, h, w] - Coil sensitivity phases (optional)

    Returns:
        Dict {"signal": Tensor} with shape [b, FAs, TEs, coils, d, h, w] or [b, FAs, TEs, d, h, w]
    """

    def __init__(
        self,
        flip_angles: Iterable[float],
        echo_times: Iterable[float],
        TR: float,
        voxel_dimensions_mm: Optional[Iterable[float]] = None,
        intra_voxel_dephasing: bool = False,
        num_coils: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.flip_angles = torch.tensor(
            list(flip_angles), dtype=torch.float32, device=device
        )
        self.echo_times = torch.tensor(
            list(echo_times), dtype=torch.float32, device=device
        )
        self.alphas_rad = self.flip_angles * torch.pi / 180.0
        self.TR = TR
        self.num_flip_angles = len(self.flip_angles)
        self.num_echo_times = len(self.echo_times)
        self.num_coils = num_coils
        self.device = device

        # Initialize component operators
        self.t1_operator = T1_NonLinearOperator(
            flip_angles=flip_angles, TR=TR, device=device
        )
        # Proton density operator: multiplies E1 by M -> returns key 'S'
        self.proton_op = ProtonDensityOperator(signal_key="E1", device=device)

        # T2* with field correction operator that accepts pre-computed ME1
        # Use the modified variant which expects 'ME1' in the input dict
        # Pass the raw lists provided by the caller (not tensors)
        self.t2star_field_op = MultiFA_T2star_Field_Correction_NonLinearOperator(
            FAs=flip_angles, TEs=echo_times, device=device
        )

        # Intravoxel static dephasing attenuation operator
        if intra_voxel_dephasing:
            self.intravoxel_dephasing_op = LinearGradientIntravoxelDephasing(
                TEs=echo_times, voxel_dimensions_mm=voxel_dimensions_mm, device=device
            )

        # Initial phase correction operator to expand/apply coil phases when needed
        # Configure to support per-channel phase initialization (for multi-coil)
        self.init_phase_op = Init_Phase_Correction(
            device=device,
            phase_init_for_each_channel=False,
            phase_init_for_each_FA=True,
            # phase_init_for_each_FA=False,
            # global_phase_init=True,
        )

        if device is not None:
            self.to(device)

    def update_parameters(
        self,
        flip_angles: Optional[Iterable[float]] = None,
        echo_times: Optional[Iterable[float]] = None,
        TR: Optional[float] = None,
        voxel_dimensions_mm: Optional[Iterable[float]] = None,
        num_coils: Optional[int] = None,
    ):
        """Update sequence parameters."""
        if flip_angles is not None:
            self.flip_angles = torch.tensor(
                list(flip_angles), dtype=torch.float32, device=self.device
            )
            self.alphas_rad = self.flip_angles * torch.pi / 180.0
            self.num_flip_angles = len(self.flip_angles)
            self.t1_operator.update_parameters(flip_angles=flip_angles)
            # propagate to proton and t2 operators
            if hasattr(self, "proton_op"):
                self.proton_op.signal_key = "E1"
            if hasattr(self, "t2star_field_op"):
                self.t2star_field_op.update_parameters(FAs=flip_angles)

        if echo_times is not None:
            self.echo_times = torch.tensor(
                list(echo_times), dtype=torch.float32, device=self.device
            )
            self.num_echo_times = len(self.echo_times)
            if hasattr(self, "t2star_field_op"):
                self.t2star_field_op.update_parameters(TEs=echo_times)
            if hasattr(self, "intravoxel_dephasing_op"):
                self.intravoxel_dephasing_op.update_parameters(TEs=echo_times)
            if hasattr(self, "init_phase_op") and hasattr(
                self.init_phase_op, "update_parameters"
            ):
                try:
                    self.init_phase_op.update_parameters(TEs=echo_times)
                except Exception:
                    pass

        if TR is not None:
            self.TR = TR
            self.t1_operator.update_parameters(TR=TR)

        if voxel_dimensions_mm is not None and hasattr(self, "intravoxel_dephasing_op"):
            self.intravoxel_dephasing_op.update_parameters(
                voxel_dimensions_mm=voxel_dimensions_mm
            )

        if num_coils is not None:
            self.num_coils = num_coils

    def _compute_delta_f_gradient(self, delta_f: Tensor) -> Tensor:
        voxel_dims = getattr(self.intravoxel_dephasing_op, "voxel_dimensions", None)
        if voxel_dims is None:
            raise ValueError(
                "Voxel dimensions are required to compute Delta_f gradients."
            )
        spacing = tuple(float(dim.item()) for dim in voxel_dims)
        grads = torch.gradient(delta_f, spacing=spacing, dim=(-3, -2, -1))
        return torch.stack(grads, dim=1)

    def A(
        self, x: Dict[str, Tensor], apply_init_phase_op: bool = True, **kwargs
    ) -> Dict[str, Tensor]:
        """
        Forward operator implementing the full MRI signal model.
        """
        # Step 1: Compute Ernst angle factor E1(R1) using T1 operator
        e1_result = self.t1_operator.A(x)
        E1 = e1_result["E1"]  # Shape: [b, FAs, d, h, w]

        # Step 2: Use ProtonDensityOperator to compute ME1 = M * E1
        if "M" not in x:
            raise ValueError("MRI_Signal_Model_Operator requires 'M' in input dict.")

        # Prepare a small dict for the proton operator that expects keys 'M' and signal_key
        pd_input = {"M": x["M"], "E1": E1}
        pd_out = self.proton_op.A(pd_input)
        # ProtonDensityOperator returns key 'S' (the scaled signal). We'll treat it as ME1.
        ME1 = pd_out["S"]  # shape [b, FAs, d, h, w]

        # Step 3-5: Delegate T2* decay and field-correction to modified T2* operator
        # The modified operator expects 'ME1', 'T2star' or 'R2star', and 'Delta_f'
        t2_input = {"ME1": ME1}
        if "T2star" in x:
            t2_input["T2star"] = x["T2star"]
        if "R2star" in x:
            t2_input["R2star"] = x["R2star"]
        if "Delta_f" in x:
            t2_input["Delta_f"] = x["Delta_f"]
        else:
            raise ValueError(
                "MRI_Signal_Model_Operator requires 'Delta_f' in input dict."
            )

        t2_out = self.t2star_field_op.A(t2_input)
        # t2_out['signal'] has shape [b, FAs, TEs, d, h, w] (or with singleton coil dim)
        signal_with_field = t2_out["signal"]

        # Step 6: Apply intravoxel static dephasing attenuation when gradient is provided
        # intravoxel_keys = getattr(
        #     LinearGradientIntravoxelDephasing, "_GRADIENT_KEYS", tuple()
        # )
        if hasattr(self, "intravoxel_dephasing_op"):
            gradient_tensor = self._compute_delta_f_gradient(x["Delta_f"])
            ce_factors = self.intravoxel_dephasing_op.A(
                {"Delta_f_grad": gradient_tensor}
            )["C_e"]
            signal_with_field = signal_with_field * ce_factors.unsqueeze(1)

        # Step 7: Apply initial coil phase / expand coil dimension when configured
        init_input = {"signal": signal_with_field}
        if "phi_0" in x and x["phi_0"] is not None:
            init_input["phi_0"] = x["phi_0"]
        if apply_init_phase_op:
            final_signal = self.init_phase_op.A(init_input)
        else:
            final_signal = signal_with_field

        # Convert to complex if input was real
        if not final_signal.dtype.is_complex:
            final_signal = final_signal.to(
                torch.complex64 if x["M"].dtype == torch.float32 else torch.complex128
            )

        return {"signal": final_signal}
