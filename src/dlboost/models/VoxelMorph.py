import torch
import torch.nn as nn

# import torch.nn.functional as nnf
from torch.distributions.normal import Normal

from .building_blocks import ConvBlock
from .SpatialTransformNetwork import SpatialTransformNetwork

####################################
# Registration Neutral Network
####################################

# Some implementation here is adopted from VoxelMorph.


# noinspection PyUnresolvedReferences
class unet_core(nn.Module):
    """
    [unet_core] is a class representing the U-Net implementation that takes in
    a fixed image and a moving image and outputs a flow-field
    """

    def __init__(self, dim, enc_nf, dec_nf, full_size=True):
        """
        Instiatiate UNet model
            :param dim: dimension of the image passed into the net
            :param enc_nf: the number of features maps in each layer of encoding stage
            :param dec_nf: the number of features maps in each layer of decoding stage
            :param full_size: boolean value representing whether full amount of decoding
                            layers
        """
        super().__init__()

        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7

        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = 2 if i == 0 else enc_nf[i - 1]
            self.enc.append(ConvBlock(dim, prev_nf, enc_nf[i], 2))

        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(ConvBlock(dim, enc_nf[-1], dec_nf[0]))  # 1
        self.dec.append(ConvBlock(dim, dec_nf[0] * 2, dec_nf[1]))  # 2
        self.dec.append(ConvBlock(dim, dec_nf[1] * 2, dec_nf[2]))  # 3
        self.dec.append(ConvBlock(dim, dec_nf[2] + enc_nf[0], dec_nf[3]))  # 4
        self.dec.append(ConvBlock(dim, dec_nf[3], dec_nf[4]))  # 5

        if self.full_size:
            self.dec.append(ConvBlock(dim, dec_nf[4] + 2, dec_nf[5], 1))

        if self.vm2:
            self.vm2_conv = ConvBlock(dim, dec_nf[5], dec_nf[6])

        self.upsample = nn.Upsample(
            scale_factor=2 if dim == 2 else (1, 2, 2), mode="nearest"
        )

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and moving image
        """
        # Get encoder activations
        x_enc = [x]
        for l in self.enc:
            x_enc.append(l(x_enc[-1]))

        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
            y = self.dec[i](y)
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)

        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)

        # Upsample to full res, concatenate and conv
        if self.full_size:
            y = self.upsample(y)
            y = torch.cat([y, x_enc[0]], dim=1)
            y = self.dec[5](y)

        # Extra conv for vm2
        if self.vm2:
            y = self.vm2_conv(y)

        return y


class VoxelMorph(nn.Module):
    """
    [cvpr2018_net] is a class representing the specific implementation for
    the 2018 implementation of voxelmorph.
    """

    def __init__(self, vol_size, enc_nf, dec_nf, full_size=True):
        """
        Instiatiate 2018 model
            :param vol_size: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
            :param full_size: boolean value full amount of decoding layers
        """
        super().__init__()

        dim = len(vol_size)

        self.unet_model = unet_core(dim, enc_nf, dec_nf, full_size)

        # One conv to get the flow field
        conv_fn = getattr(nn, "Conv%dd" % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)

        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(
            nd.sample(self.flow.weight.shape), requires_grad=True
        )
        self.flow.bias = nn.Parameter(
            torch.zeros(self.flow.bias.shape), requires_grad=True
        )

        self.spatial_transform = SpatialTransformNetwork(vol_size)  # , dims=dim)

    def forward(self, src, tgt, to_warp=None):
        """
        Pass input x through forward once
            :param src: moving image that we want to shift
            :param tgt: fixed image that we want to shift to
        """
        x = torch.cat([src, tgt], dim=1)
        x = self.unet_model(x)
        flow = self.flow(x)
        # if to_warp is None:
        y = self.spatial_transform(src, flow)
        # y = self.spatial_transform(to_warp, flow)

        return y, flow
