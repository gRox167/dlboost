import einx
import torch
from deepinv.physics import DecomposablePhysics


class LinearFittingOperator(DecomposablePhysics):
    """
    Linear fitting operator that models y = mx + b for 3D images.

    Processes one slope image and multiple intercept images in the channel dimension.
    The linear fitting is performed using FA (Flip Angles) and TE (Echo Times) parameters.
    """

    def __init__(
        self,
        design_matrix,
        device="cpu",
        **kwargs,
    ):
        """
        Initialize the linear fitting operator.

        Args:
            FA_values: torch.Tensor or array-like - Flip Angles for linear fitting
            TE_values: torch.Tensor or array-like - Echo Times for linear fitting
            device: str - Device to use for computations
        """
        super().__init__(**kwargs)
        self.device = device
        self.update_parameters(design_matrix, device)

    def update_parameters(self, design_matrix=None, device=None, **kwargs):
        """
        Update the FA and TE parameters of the model.

        Args:
            FA_values: torch.Tensor or array-like - Flip Angles
            TE_values: torch.Tensor or array-like - Echo Times
            device: str - Device to use for computations
        """
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
        self.V_tensor = V  # Shape: [1+n_intercepts, 1+n_intercepts]

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
                Shape: [batch_size, 1+n_intercepts, height, width, depth]

        Returns:
            torch.Tensor - Result of applying V
                Shape: [batch_size, 1+n_intercepts, height, width, depth]
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
                Shape: [batch_size, 1+n_intercepts, height, width, depth]

        Returns:
            torch.Tensor - Result of applying V^H
                Shape: [batch_size, 1+n_intercepts, height, width, depth]
        """
        return einx.dot(
            "n1 n2, b n1 d h w -> b n2 d h w",
            self.V_tensor,
            x,
        )
