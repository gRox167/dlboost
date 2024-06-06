__all__ = ["ComplexUnet", "ComplexUnetDenoiser", "ComplexUnet_norm"]


from collections.abc import Sequence

import torch.nn as nn
import torch
from torch import Tensor, vmap
# from einops import rearrange,repeat, reduce

from monai.apps.reconstruction.networks.nets.utils import (
    complex_normalize,
    divisible_pad_t,
    inverse_divisible_pad_t,
    reshape_channel_complex_to_last_dim,
    reshape_complex_to_channel_dim,
)
from monai.networks.nets.basic_unet import BasicUNet

def percentile(t: torch.tensor, l, h):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
       
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    l_ = 1 + round(.01 * float(l) * (t.numel() - 1))
    h_ = 1 + round(.01 * float(h) * (t.numel() - 1))
    l_th = t.kthvalue(l_).values
    h_th = t.kthvalue(h_).values
    return l_th, h_th

# percentile_v = vmap(percentile, in_dims=(0, None, None), out_dims=0)
# clamp_v = vmap(torch.clamp, in_dims=(0, 0, 0), out_dims=0)
def complex_normalize_abs_95(x, start_dim = 0):
    x_abs = x.abs()  
    min_95, max_95 = percentile(x_abs.flatten(start_dim), 2.5, 97.5)
    x_abs_clamped = torch.clamp(x_abs, min_95, max_95)
    # x_abs_clamped = x_abs.clamp(min=min_95, max=max_95)
    mean = torch.mean(x_abs_clamped)
    std = torch.std(x_abs_clamped, unbiased=False)
    return (x - mean) / std, mean.expand_as(x_abs_clamped), std.expand_as(x_abs_clamped)
complex_normalize_abs_95_v = vmap(complex_normalize_abs_95, in_dims= 0, out_dims=( 0,0,0 ))

class ComplexUnet(nn.Module):
    def __init__(
        self,
        in_channels:int = 3, 
        out_channels:int = 3,
        spatial_dims: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "nontrainable",
        pad_factor: int = 16,
        conv_net: nn.Module | None = None,
    ):
        super().__init__()
        self.unet: nn.Module
        self.in_channels = in_channels
        if conv_net is None:
            self.unet = BasicUNet(
                spatial_dims=spatial_dims,
                in_channels=2*in_channels,
                out_channels=2*out_channels,
                features=features,
                act=act,
                norm=norm,
                bias=bias,
                dropout=dropout,
                upsample=upsample,
            )
        else:
            # assume the first layer is convolutional and
            # check whether in_channels == 2
            # params = [p.shape for p in conv_net.parameters()]
            # if params[0][1] != 2:
            #     raise ValueError(f"in_channels should be 2 but it's {params[0][1]}.")
            self.unet = conv_net

        self.pad_factor = pad_factor

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input of shape (B,C,H,W) for 2D data or (B,C,H,W,D) for 3D data

        Returns:
            output of shape (B,C,H,W) for 2D data or (B,C,H,W,D) for 3D data
        """
        # suppose the input is 2D, the comment in front of each operator below shows the shape after that operator
        # print(x.shape)
        x, mean, std = complex_normalize_abs_95_v(x)  # x will be of shape (B,C*2,H,W)

        x = torch.view_as_real(x)

        x = reshape_complex_to_channel_dim(x)  # x will be of shape (B,C*2,H,W)
        # x, mean, std = complex_normalize(x)  # x will be of shape (B,C*2,H,W)
        x = self.unet(x)
        x = reshape_channel_complex_to_last_dim(x)  # x will be of shape (B,C,H,W,2)
        x = torch.view_as_complex(x.contiguous())
        x = x * std + mean
        return x

    
class ComplexUnet_norm(nn.Module):
    def __init__(
        self,
        in_channels:int = 3, 
        out_channels:int = 3,
        spatial_dims: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "nontrainable",
        pad_factor: int = 16,
        conv_net: nn.Module | None = None,
    ):
        super().__init__()
        self.unet: nn.Module
        self.in_channels = in_channels
        if conv_net is None:
            self.unet = BasicUNet(
                spatial_dims=spatial_dims,
                in_channels=2*in_channels,
                out_channels=2*out_channels,
                features=features,
                act=act,
                norm=norm,
                bias=bias,
                dropout=dropout,
                upsample=upsample,
            )
        else:
            # assume the first layer is convolutional and
            # check whether in_channels == 2
            # params = [p.shape for p in conv_net.parameters()]
            # if params[0][1] != 2:
            #     raise ValueError(f"in_channels should be 2 but it's {params[0][1]}.")
            self.unet = conv_net

        self.pad_factor = pad_factor

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input of shape (B,C,H,W) for 2D data or (B,C,H,W,D) for 3D data

        Returns:
            output of shape (B,C,H,W) for 2D data or (B,C,H,W,D) for 3D data
        """
        # suppose the input is 2D, the comment in front of each operator below shows the shape after that operator
        # print(x.shape)
        x, mean, std = complex_normalize_abs_95_v(x)  # x will be of shape (B,C*2,H,W)

        x = torch.view_as_real(x)

        x = reshape_complex_to_channel_dim(x)  # x will be of shape (B,C*2,H,W)
        # x, mean, std = complex_normalize(x)  # x will be of shape (B,C*2,H,W)

        x = self.unet(x)
        x = reshape_channel_complex_to_last_dim(x)  # x will be of shape (B,C,H,W,2)
        x = torch.view_as_complex(x.contiguous())
        x = x * std + mean
        return x


class ComplexUnetDenoiser(nn.Module):
    def __init__(
        self,
        in_channels:int = 3, 
        out_channels:int = 3,
        spatial_dims: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
        pad_factor: int = 16,
        conv_net: nn.Module | None = None,
    ):
        super().__init__()
        self.unet: nn.Module
        if conv_net is None:
            self.unet = BasicUNet(
                spatial_dims=spatial_dims,
                in_channels=2*in_channels,
                out_channels=2*out_channels,
                features=features,
                act=act,
                norm=norm,
                bias=bias,
                dropout=dropout,
                upsample=upsample,
            )
        else:
            # assume the first layer is convolutional and
            # check whether in_channels == 2
            # params = [p.shape for p in conv_net.parameters()]
            # if params[0][1] != 2:
            #     raise ValueError(f"in_channels should be 2 but it's {params[0][1]}.")
            self.unet = conv_net

        self.pad_factor = pad_factor


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
        x_ = inverse_divisible_pad_t(x_, padding_sizes)  # x will be of shape (B,C*2,H,W)

        x_ = x_ * std + mean
        x_ = reshape_channel_complex_to_last_dim(x_).contiguous()  # x will be of shape (B,C,H,W,2)
        return torch.view_as_complex(x_)
    
