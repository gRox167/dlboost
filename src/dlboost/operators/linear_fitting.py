from typing import Sequence

import einx
import torch
from deepinv.physics import DecomposablePhysics


class LinearFittingOperator(DecomposablePhysics):
    def __init__(
        self,
        design_matrix,
        device="cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.device = device
        self.design_matrix = design_matrix.to(device)
        self._compute_svd_components()

    def update_parameters(self, design_matrix=None, device=None, **kwargs):
        if design_matrix is not None:
            if not isinstance(design_matrix, torch.Tensor):
                self.design_matrix = torch.tensor(design_matrix, dtype=torch.float32)
            else:
                self.design_matrix = design_matrix

            if device is not None:
                self.design_matrix = self.design_matrix.to(device)

        if hasattr(self, "design_matrix"):
            self._compute_svd_components()

    def _compute_svd_components(self):
        """
        Compute the SVD components (U, Sigma, V) for the linear fitting operation.
        """
        # Compute the SVD components
        U, S, V = torch.svd(self.design_matrix)

        # Store the components
        self.U_tensor = U  # Shape: [n_measurements, n_measurements]
        self.mask = einx.rearrange("s -> 1 s 1 1 1", S)
        self.V_tensor = V  # Shape: [n_parameters, n_parameters]

    def U(self, x):
        """
        Apply the U matrix operation to input tensor.

        Args:
            x: torch.Tensor - Input tensor
                Shape: [batch_size, n_measurements, height, width, depth]

        Returns:
            torch.Tensor - Result of applying U
                Shape: [batch_size, n_measurements, height, width, depth]
        """
        return einx.dot(
            "m1 m2, b m2 d h w -> b m1 d h w",
            self.U_tensor,
            x,
        )

    def U_adjoint(self, x):
        """
        Apply the U^H (conjugate transpose of U) operation to input tensor.

        Args:
            x: torch.Tensor - Input tensor
                Shape: [batch_size, n_measurements, height, width, depth]

        Returns:
            torch.Tensor - Result of applying U^H
                Shape: [batch_size, n_measurements, height, width, depth]
        """
        return einx.dot(
            "m1 m2, b m1 d h w -> b m2 d h w",
            self.U_tensor,
            x,
        )

    def V(self, x):
        """
        Apply the V matrix operation to input tensor.

        Args:
            x: torch.Tensor - Input tensor
                Shape: [batch_size, n_parameters, height, width, depth]

        Returns:
            torch.Tensor - Result of applying V
                Shape: [batch_size, n_parameters, height, width, depth]
        """
        return einx.dot(
            "n1 n2, b n2 d h w -> b n1 d h w",
            self.V_tensor,
            x,
        )

    def V_adjoint(self, x):
        """
        Apply the V^H (conjugate transpose of V) operation to input tensor.

        Args:
            x: torch.Tensor - Input tensor
                Shape: [batch_size, n_parameters, height, width, depth]

        Returns:
            torch.Tensor - Result of applying V^H
                Shape: [batch_size, n_parameters, height, width, depth]
        """
        return einx.dot(
            "n1 n2, b n1 d h w -> b n2 d h w",
            self.V_tensor,
            x,
        )


def build_design_matrix(
    flip_angles: torch.Tensor | Sequence[float],
    echo_times: torch.Tensor | Sequence[float],
    with_intercept: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Build a design matrix for linear fitting based on flip angles and echo times.

    This function creates a design matrix X for linear regression that models phase
    evolution over echo times. The matrix structure depends on the with_intercept parameter:

    - If with_intercept=True: Models phase = common_slope*TE + FA_specific_bias
      Matrix shape: (num_fa * num_echoes, num_fa + 1)
      First column: echo times (common slope term)
      Remaining columns: one-hot encoding for FA-specific bias terms

    - If with_intercept=False: Models phase = FA_specific_slope*TE
      Matrix shape: (num_fa * num_echoes, num_fa)
      Each column represents a different flip angle with corresponding echo times

    Args:
        flip_angles (torch.Tensor or Sequence[float]): Flip angles for linear fitting.
        echo_times (torch.Tensor or Sequence[float]): Echo times for linear fitting.
        with_intercept (bool): Whether to include intercept terms in the design matrix.
            If True, uses a common slope with FA-specific biases.
            If False, uses FA-specific slopes with no bias.
        device (str): Device to create the tensor on.

    Returns:
        torch.Tensor: Design matrix with shape:
            - (num_fa * num_echoes, num_fa + 1) if with_intercept=True
            - (num_fa * num_echoes, num_fa) if with_intercept=False

    Example:
        For 2 flip angles [8, 15] and 3 echo times [2, 4, 6] with with_intercept=True:
        X shape: (6, 3) where each row corresponds to [TE, FA8_bias, FA15_bias]
        Row 0: [2, 1, 0] (FA8, TE1)
        Row 1: [4, 1, 0] (FA8, TE2)
        Row 2: [6, 1, 0] (FA8, TE3)
        Row 3: [2, 0, 1] (FA15, TE1)
        Row 4: [4, 0, 1] (FA15, TE2)
        Row 5: [6, 0, 1] (FA15, TE3)
    """
    print(flip_angles, echo_times)
    num_fa, num_echoes = len(flip_angles), len(echo_times)
    n_samples = num_fa * num_echoes

    # Determine number of parameters
    n_params = (
        num_fa + 1 if with_intercept else num_fa
    )  # FA specific slopes only if not with_intercept

    # Initialize design matrix
    X = torch.zeros((n_samples, n_params), dtype=torch.float32, device=device)
    print(f"Design matrix shape: {X.shape}")

    # Fill design matrix
    for fa_idx in range(num_fa):
        for echo_idx in range(num_echoes):
            sample_idx = fa_idx * num_echoes + echo_idx
            if with_intercept:
                # First column: echo times (common slope)
                X[sample_idx, 0] = echo_times[echo_idx]
                # FA-specific bias column
                X[sample_idx, 1 + fa_idx] = 1.0
            else:
                # FA-specific slope with echo time
                X[sample_idx, fa_idx] = echo_times[echo_idx]
    return X


def build_target_matrix(
    multi_flip_angle_multi_echo_images: torch.Tensor,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Build a target matrix from multi-flip-angle multi-echo images for linear regression.

    This function reshapes the input image data into a format suitable for linear
    regression analysis. The output matrix Y has the same sample ordering as the
    design matrix created by build_design_matrix().

    Args:
        multi_flip_angle_multi_echo_images (torch.Tensor): Image data with shape
            (num_fa, num_echoes, D, H, W) where:
            - num_fa: number of flip angles
            - num_echoes: number of echo times
            - D, H, W: spatial dimensions (depth, height, width)
        flip_angles (torch.Tensor or Sequence[float], optional): Flip angles corresponding
            to the data. Used for validation but not required for matrix construction.
        echo_times (torch.Tensor or Sequence[float], optional): Echo times corresponding
            to the data. Used for validation but not required for matrix construction.
        device (str): Device to create the tensor on.

    Returns:
        torch.Tensor: Target matrix with shape (1, n_samples, D, H, W) where:
            - n_samples = num_fa * num_echoes
            - Sample ordering matches build_design_matrix():
              [FA1_TE1, FA1_TE2, ..., FA1_TEn, FA2_TE1, FA2_TE2, ..., FAm_TEn]

    Example:
        For input shape (2, 3, 64, 64, 64) with 2 flip angles and 3 echo times:
        Output shape: (1, 6, 64, 64, 64)
        Sample 0: FA1, TE1
        Sample 1: FA1, TE2
        Sample 2: FA1, TE3
        Sample 3: FA2, TE1
        Sample 4: FA2, TE2
        Sample 5: FA2, TE3
    """
    # Validate input dimensions
    if multi_flip_angle_multi_echo_images.ndim != 5:
        raise ValueError(
            f"Expected 5D input tensor with shape (num_fa, num_echoes, D, H, W), "
            f"got {multi_flip_angle_multi_echo_images.ndim}D tensor with shape "
            f"{multi_flip_angle_multi_echo_images.shape}"
        )

    num_fa, num_echoes, D, H, W = multi_flip_angle_multi_echo_images.shape
    n_samples = num_fa * num_echoes

    # Initialize target matrix
    # Shape: (1, n_samples, D, H, W) - batch dimension of 1 for compatibility
    Y = torch.zeros(
        (1, n_samples, D, H, W),
        device=device,
        dtype=multi_flip_angle_multi_echo_images.dtype,
    )

    # Fill target matrix with same ordering as design matrix
    sample_idx = 0
    for fa_idx in range(num_fa):
        for echo_idx in range(num_echoes):
            Y[0, sample_idx] = multi_flip_angle_multi_echo_images[fa_idx, echo_idx]
            sample_idx += 1

    return Y


def build_regression_matrices(
    multi_flip_angle_multi_echo_images=None,
    flip_angles=[8, 15],
    echo_times=[4, 8, 12],
    with_intercept=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Build design and target matrices for linear regression on multi-flip-angle multi-echo data.

    This function constructs matrices for performing linear regression on phase data
    across multiple flip angles and echo times. It combines the functionality of
    build_design_matrix() and build_target_matrix() for convenience.

    The design matrix X models phase evolution over time with two possible structures:
    - with_intercept=True: phase = common_slope*TE + FA_specific_bias
    - with_intercept=False: phase = FA_specific_slope*TE

    Args:
        multi_flip_angle_multi_echo_images (torch.Tensor, optional): Image data with shape
            (num_fa, num_echoes, D, H, W). If None, only returns design matrix.
        flip_angles (list or torch.Tensor): Flip angles corresponding to the data.
        echo_times (list or torch.Tensor): Echo times corresponding to each echo.
        with_intercept (bool): Whether to include intercept terms in the linear model.
            If True, fits a common slope with FA-specific biases.
            If False, fits FA-specific slopes with no bias terms.
        device (str): Computation device ('cuda' or 'cpu').

    Returns:
        tuple: (X, Y) where:
            - X (torch.Tensor): Design matrix for linear regression
            - Y (torch.Tensor): Target matrix (None if multi_flip_angle_multi_echo_images is None)

    Raises:
        ValueError: If input tensor dimensions don't match provided flip_angles/echo_times.

    Example:
        >>> # Create synthetic data
        >>> images = torch.randn(2, 3, 32, 32, 32)  # 2 FAs, 3 TEs
        >>> X, Y = build_regression_matrices(
        ...     images,
        ...     flip_angles=[8, 15],
        ...     echo_times=[2, 4, 6],
        ...     with_intercept=True
        ... )
        >>> print(X.shape, Y.shape)  # (6, 3), (1, 6, 32, 32, 32)
    """
    # Build design matrix
    X = build_design_matrix(
        flip_angles=flip_angles,
        echo_times=echo_times,
        with_intercept=with_intercept,
        device=device,
    )

    # Build target matrix if image data is provided
    Y = None
    if multi_flip_angle_multi_echo_images is not None:
        Y = build_target_matrix(
            multi_flip_angle_multi_echo_images=multi_flip_angle_multi_echo_images,
            device=device,
        )

    return X, Y
