import torch
from deepinv.loss import Loss, MCLoss
from torch import Tensor
from torch.nn import MSELoss


class ImageDomainNoise2NoiseLoss(MCLoss):
    def __init__(self, metric="l2", complex_data_flag=True, **kwargs):
        super().__init__(metric, **kwargs)
        self.name = "ImageDomainNoise2Noise"
        if metric == "l2":
            self.metric = MSELoss()
        elif metric == "l1":
            self.metric = torch.nn.L1Loss()
        else:
            raise ValueError("metric must be either 'l2' or 'l1'")
        self.complex_data_flag = complex_data_flag

    def forward(
        self,
        x_net: Tensor,
        x: Tensor,
        y: Tensor,
        physics,
        model=None,
        **kwargs,
    ) -> torch.Tensor:
        if self.complex_data_flag:
            x_net = torch.view_as_real(x_net)
            x = torch.view_as_real(x)
        return self.metric(x_net, x)


class MeasurementDomainNoise2NoiseLoss(MCLoss):
    def __init__(
        self,
        metric="l2",
        weighted_flag=True,
        amplitude_weight_flag=False,
        amplitude_weight_epsilon=1e-3,
        **kwargs,
    ):
        super().__init__(metric, **kwargs)
        self.name = "MeasurementDomainNoise2NoiseLoss"
        if metric == "l2":
            metric = MSELoss()
        elif metric == "l1":
            metric = torch.nn.L1Loss()
        else:
            raise ValueError("metric must be either 'l2' or 'l1'")
        self.metric = metric
        self.weighted_flag = weighted_flag
        self.amplitude_weight_flag = amplitude_weight_flag
        self.amplitude_weight_epsilon = amplitude_weight_epsilon

    def forward(
        self,
        x_net: Tensor,
        x: Tensor | None,
        y: Tensor,
        physics,
        model=None,
        **kwargs,
    ) -> torch.Tensor:
        if self.weighted_flag:
            physics.weight = physics.weight.to(x_net.device)
            y_hat = physics.A(x_net)
            y_hat *= physics.weight
            y *= physics.weight
        else:
            y_hat = physics.A(x_net)
        if self.amplitude_weight_flag:
            y_hat_detatched = y_hat.detach().abs()
            norm_factor = y_hat_detatched.max()
            amplitude_weight = 1 / (
                y_hat_detatched / norm_factor + self.amplitude_weight_epsilon
            )
            y_hat *= amplitude_weight
            y *= amplitude_weight
        return self.metric(torch.view_as_real(y_hat), torch.view_as_real(y))


class CombinedLoss(Loss):
    def __init__(
        self,
        coeffs=[1.0, 1.0],
        losses=[ImageDomainNoise2NoiseLoss(), MeasurementDomainNoise2NoiseLoss()],
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = "CombinedLoss"
        self.coeffs = coeffs
        self.losses = losses
        self.verbose = verbose

    def forward(
        self,
        x_net: Tensor,
        x: Tensor,
        y: Tensor,
        physics,
        model,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Computes the loss.

        :param torch.Tensor x_net: Reconstructed image :math:`\inverse{y}`.
        :param torch.Tensor x: Reference image.
        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction function.

        :return: (:class:`torch.Tensor`) loss, the tensor size might be (1,) or (batch size,).
        """
        calculated_losses = [
            loss(x_net, x, y, physics, model, **kwargs) for loss in self.losses
        ]
        loss = sum(coeff * loss for coeff, loss in zip(self.coeffs, calculated_losses))
        if self.verbose:
            print(
                f"Losses: {[loss.item() for loss in calculated_losses]}, Combined Loss: {loss.item()}"
            )
        return loss
