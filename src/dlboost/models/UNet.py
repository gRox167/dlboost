# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
from re import split

import torch.nn as nn
import torch
from torch import Tensor
from einops import rearrange,repeat

from monai.apps.reconstruction.networks.nets.utils import (
    complex_normalize,
    divisible_pad_t,
    inverse_divisible_pad_t,
    reshape_channel_complex_to_last_dim,
    reshape_complex_to_channel_dim,
)
from monai.networks.nets.basic_unet import BasicUNet


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
        upsample: str = "deconv",
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
        x = torch.view_as_real(x)
        x = reshape_complex_to_channel_dim(x)  # x will be of shape (B,C*2,H,W)
        x, mean, std = complex_normalize(x)  # x will be of shape (B,C*2,H,W)
        # pad input
        # x, padding_sizes = divisible_pad_t(
        #     x, k=self.pad_factor
        # )  # x will be of shape (B,C*2,H',W') where H' and W' are for after padding

        x = self.unet(x)
        # inverse padding
        # x = inverse_divisible_pad_t(x, padding_sizes)  # x will be of shape (B,C*2,H,W)
        std = repeat(std, "b c ... -> b (c r) ...", r= x.shape[1]//(self.in_channels*2))
        mean = repeat(mean, "b c ... -> b (c r) ...", r= x.shape[1]//(self.in_channels*2))
        x = x * std + mean
        x = reshape_channel_complex_to_last_dim(x)  # x will be of shape (B,C,H,W,2)
        return torch.view_as_complex(x.contiguous())
    


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
        # breakpoint()
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
    

class UnetCVF(nn.Module):
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
        self.denoiser: nn.Module
        self.noise_seperator: nn.Module
        if conv_net is None:
            self.denoiser = ComplexUnet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                features=features,
                act=act,
                norm=norm,
                bias=bias,
                dropout=dropout,
                upsample=upsample,
            )
            self.noise_seperator = ComplexUnet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=2*out_channels,
                features=features,
                act=act,
                norm=norm,
                bias=bias,
                dropout=dropout,
                upsample=upsample,
            )
        else:
            self.unet = conv_net

        self.pad_factor = pad_factor


    def forward(self, x: Tensor) -> Tensor:
        x_noisy = x
        x_clean = self.denoiser(x_noisy)
        x_noise = x_noisy - x_clean
        x_output = self.noise_seperator(x_noise)
        # x = x_output * std.repeat(1,3,1,1,1) + mean.repeat(1,3,1,1,1)
        
        return x_clean, x_output[:,0::2,...], x_output[:,1::2,...]
