import torch
from torch import nn
from torch.nn import functional as F
from typing import Sequence, Union, Tuple, Optional
from monai.networks.blocks import UnetUpBlock, UnetResBlock, UnetBasicBlock
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.networks.layers.utils import get_act_layer, get_dropout_layer, get_norm_layer
from monai.utils import ensure_tuple_rep
from einops.layers.torch import Rearrange
from math import prod

from dlboost.models.BasicUNet import Down


class TwoConv(nn.Sequential):
    """two convolutions."""

    def __init__(
        self,
        spatial_dims: int,
        kernel_size: int,
        in_chns: int,
        out_chns: int,
        act: str | tuple,
        norm: str | tuple,
        bias: bool,
        padding: int | tuple | None = 1,
        dropout: float | tuple = 0.0,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        """
        super().__init__()

        conv_0 = Convolution(spatial_dims, in_chns, out_chns, kernel_size=kernel_size,act=act, norm=norm, dropout=dropout, bias=bias, padding=padding)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, kernel_size=kernel_size, act=act, norm=norm, dropout=dropout, bias=bias, padding=padding
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)

class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    def __init__(
        self,
        spatial_dims: int,
        kernel_size: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: str | tuple,
        norm: str | tuple,
        bias: bool,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
        pre_conv: nn.Module | str | None = "default",
        interp_mode: str = "linear",
        align_corners: bool | None = True,
        halves: bool = True,
        is_pad: bool = True,
        upsample_factors = (1,2,2)
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the encoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            pre_conv: a conv block applied before upsampling.
                Only used in the "nontrainable" or "pixelshuffle" mode.
            interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
                Only used in the "nontrainable" mode.
            align_corners: set the align_corners parameter for upsample. Defaults to True.
                Only used in the "nontrainable" mode.
            halves: whether to halve the number of channels during upsampling.
                This parameter does not work on ``nontrainable`` mode if ``pre_conv`` is `None`.
            is_pad: whether to pad upsampling features to fit features from encoder. Defaults to True.

        """
        super().__init__()
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            upsample_factors,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, kernel_size, cat_chns + up_chns, out_chns, act, norm, bias, None,dropout)
        self.is_pad = is_pad

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        """

        Args:
            x: features to be upsampled.
            x_e: optional features from the encoder, if None, this branch is not in use.
        """
        x_0 = self.upsample(x)

        if x_e is not None and torch.jit.isinstance(x_e, torch.Tensor):
            if self.is_pad:
                # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
                dimensions = len(x.shape) - 2
                sp = [0] * (dimensions * 2)
                for i in range(dimensions):
                    if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                        sp[i * 2 + 1] = 1
                x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0)

        return x

class AnisotropicUNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        strides = ((1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
        kernel_sizes = ((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        dropout: float | tuple = 0.0,
        bias = True,
        upsample: str = "nontrainable",
    ):
        super().__init__()
        # fea = ensure_tuple_rep(features, 6)
        # print(f"BasicUNet features: {fea}.")
        self.conv_0 = nn.Sequential(*[UnetResBlock(spatial_dims, in_channels, features[0], kernel_sizes[0], 1, norm, act, dropout)])
        self.down_1 =UnetBasicBlock(spatial_dims, features[0], features[1], kernel_sizes[0], strides[0], norm, act, dropout) 
        self.down_2 =UnetBasicBlock(spatial_dims, features[1], features[2], kernel_sizes[1], strides[1], norm, act, dropout) 
        self.down_3 =UnetBasicBlock(spatial_dims, features[2], features[3], kernel_sizes[2], strides[2], norm, act, dropout) 
        self.down_4 =UnetBasicBlock(spatial_dims, features[3], features[4], kernel_sizes[3], strides[3], norm, act, dropout) 

        self.upcat_4 = UpCat(spatial_dims, kernel_sizes[3], features[4], features[3], features[3], act, norm, bias, dropout, upsample, upsample_factors=strides[3])
        self.upcat_3 = UpCat(spatial_dims, kernel_sizes[2], features[3], features[2], features[2], act, norm, bias, dropout, upsample, upsample_factors=strides[2])
        self.upcat_2 = UpCat(spatial_dims, kernel_sizes[1], features[2], features[1], features[1], act, norm, bias, dropout, upsample, upsample_factors=strides[1])
        self.upcat_1 = UpCat(spatial_dims, kernel_sizes[0], features[1], features[0], features[5], act, norm, bias, dropout, upsample, halves=False, upsample_factors=strides[0])
        # self.bridge_0 = nn.Sequential(*[UnetResBlock(spatial_dims, features[0], features[0], kernel_sizes[0], 1, norm, act, dropout) for _ in range(1)])
        # self.refine_conv = nn.Sequential(*[UnetResBlock(spatial_dims, features[0], features[0], (kernel_sizes[0]), 1, norm, act, dropout) for _ in range(2)])
        self.final_conv = UnetResBlock(spatial_dims, features[0], out_channels, 1, 1, norm, act, dropout)
        # self.final_conv = Conv["conv", spatial_dims](features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N-1])``, N is defined by `spatial_dims`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N-1])``.
        """
        x0 = self.conv_0(x)

        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)

        # x0 = self.bridge_0(x0)
        u1 = self.upcat_1(u2, x0)

        # u1 = self.refine_conv(u1)
        logits = self.final_conv(u1)
        return logits

