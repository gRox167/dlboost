__all__ = ["ComplexUnet", "ComplexUnetDenoiser", "ComplexUnet_norm"]


from collections.abc import Sequence

import einx
import torch
import torch.nn as nn

# from einops import rearrange,repeat, reduce
from monai.apps.reconstruction.networks.nets.utils import (
    complex_normalize,
    divisible_pad_t,
    inverse_divisible_pad_t,
    reshape_channel_complex_to_last_dim,
    reshape_complex_to_channel_dim,
)
from monai.networks.nets.basic_unet import BasicUNet
from torch import Tensor, vmap

from dlboost.utils.tensor_utils import complex_normalize_abs_95

complex_normalize_abs_95_v = vmap(
    complex_normalize_abs_95, in_dims=0, out_dims=(0, 0, 0)
)


class ComplexUnet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        spatial_dims: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "nontrainable",
        pad_factor: int = 16,
        conv_net: nn.Module | None = None,
        input_append_channel: int = 0,
        norm_with_given_std: bool = False,
        residual_wrapper: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.input_append_channel = input_append_channel
        self.norm_with_given_std = norm_with_given_std
        self.pad_factor = pad_factor
        self.residual_wrapper = residual_wrapper
        if conv_net is None:
            self.unet = BasicUNet(
                spatial_dims=spatial_dims,
                in_channels=2 * in_channels + input_append_channel,
                out_channels=2 * out_channels,
                features=features,
                act=act,
                norm=norm,
                bias=bias,
                dropout=dropout,
                upsample=upsample,
            )
        else:
            self.unet = conv_net

    def forward(
        self,
        x: Tensor,
        std: Tensor | float = 1.0,
        input_append_channel: Tensor | None = None,
    ) -> Tensor:
        """
        Process complex tensor through UNet architecture.

        Args:
            x: Complex input of shape (B,C,H,W) for 2D data or (B,C,H,W,D) for 3D data
            std: Standard deviation for normalization, either provided or calculated
            input_append_channel: Optional additional channel to append to input

        Returns:
            Complex tensor of shape (B,C,H,W) for 2D data or (B,C,H,W,D) for 3D data
        """

        # input_device = x.device
        # model_device = next(self.unet.parameters()).device
        # if input_device != model_device:
        #     x.to(model_device)

        # Normalize input
        if not self.norm_with_given_std:
            _, std = complex_normalize_abs_95_v(x)
        x = x / std

        # Convert to real representation and prepare for UNet
        x = reshape_complex_to_channel_dim(torch.view_as_real(x))

        # Append additional channels if provided
        if input_append_channel is not None:
            x = einx.rearrange(
                "b c1 ..., b c2 ... -> b (c1+c2) ...", x, input_append_channel
            )

        # Process through UNet with padding
        x, pad_sizes = divisible_pad_t(x, self.pad_factor)
        if self.residual_wrapper:
            input_tensor = (
                x[:, : -input_append_channel.shape[1]]
                if input_append_channel is not None
                else x
            )
            x = input_tensor + self.unet(x)
        else:
            x = self.unet(x)
        x = inverse_divisible_pad_t(x, pad_sizes)

        # Convert back to complex representation
        x = torch.view_as_complex(
            reshape_channel_complex_to_last_dim(x).contiguous().to(dtype=torch.float32)
        )

        # Apply denormalization
        x *= std
        return x


class ComplexUnetDenoiser(ComplexUnet):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input of shape (B,C,H,W) for 2D data or (B,C,H,W,D) for 3D data

        Returns:
            output of shape (B,C,H,W) for 2D data or (B,C,H,W,D) for 3D data
        """
        # print(x.shape)
        # suppose the input is 2D, the comment in front of each operator below shows the shape after that operator
        x = torch.view_as_real(x)
        x = reshape_complex_to_channel_dim(x)  # x will be of shape (B,C*2,H,W)
        x, mean, std = complex_normalize(x)  # x will be of shape (B,C*2,H,W)
        # pad input
        x, padding_sizes = divisible_pad_t(
            x, k=self.pad_factor
        )  # x will be of shape (B,C*2,H',W') where H' and W' are for after padding
        identity = x
        x_ = self.unet(x)
        x_ -= identity
        # inverse padding
        x_ = inverse_divisible_pad_t(
            x_, padding_sizes
        )  # x will be of shape (B,C*2,H,W)

        x_ = x_ * std + mean
        x_ = reshape_channel_complex_to_last_dim(
            x_
        ).contiguous()  # x will be of shape (B,C,H,W,2)
        return torch.view_as_complex(x_)
