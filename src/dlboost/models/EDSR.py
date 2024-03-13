from math import prod
from random import shuffle

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from monai.apps.reconstruction.networks.nets.utils import (
    complex_normalize,
    divisible_pad_t,
    inverse_divisible_pad_t,
    reshape_channel_complex_to_last_dim,
    reshape_complex_to_channel_dim,
)

# import torch.nn.functional as nnf
from .building_blocks import ResBlock, activation_fn


class EDSR(nn.Module):
    def __init__(
        self, dimension, n_resblocks, n_feats, in_channels=1, out_channels=1, act="gelu"
    ):
        super().__init__()

        if dimension == 2:
            conv_fn = nn.Conv2d
        elif dimension == 3:
            conv_fn = nn.Conv3d
        else:
            raise ValueError()
        m_head = [conv_fn(in_channels, n_feats, 3, padding=3 // 2)]

        m_body = [
            ResBlock(
                dimension,
                n_feats,
                3,
                act=activation_fn[act](),  # , res_scale=res_scale
            )
            for _ in range(n_resblocks)
        ]

        m_body.append(conv_fn(n_feats, n_feats, 3, padding=3 // 2))

        m_tail = [conv_fn(n_feats, out_channels, 3, padding=3 // 2)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = reshape_complex_to_channel_dim(
            torch.view_as_real(x)
        )  # x will be of shape (B,C*2,H,W)
        x, mean, std = complex_normalize(x)  # x will be of shape (B,C*2,H,W)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        x = x * std + mean
        x = reshape_channel_complex_to_last_dim(x)  # x will be of shape (B,C,H,W,2)
        return torch.view_as_complex(x.contiguous())


class ShuffleEDSR(nn.Module):
    def __init__(
        self,
        dimension,
        n_resblocks,
        n_feats,
        down_sample_rate=[1, 2, 2],
        in_channels=1,
        out_channels=1,
        act="gelu",
    ):
        super().__init__()

        if dimension == 2:
            conv_fn = nn.Conv2d
            shuffle = Rearrange("b c  (h h2) (w w2) -> b (c h2 w2) d h w")
            unshuffle = Rearrange(
                "b (c h2 w2) d h w -> b c (h h2) (w w2)",
                h2=down_sample_rate[1],
                w2=down_sample_rate[2],
            )
        elif dimension == 3:
            conv_fn = nn.Conv3d
            shuffle = Rearrange(
                "b c (d d2) (h h2) (w w2) -> b (d2 h2 w2 c) d h w",
                d2=down_sample_rate[0],
                h2=down_sample_rate[1],
                w2=down_sample_rate[2],
            )
            unshuffle = Rearrange(
                "b (d2 h2 w2 c) d h w -> b c (d d2) (h h2) (w w2)",
                d2=down_sample_rate[0],
                h2=down_sample_rate[1],
                w2=down_sample_rate[2],
            )
        else:
            raise ValueError()
        m_head = [
            shuffle,
            conv_fn(in_channels * prod(down_sample_rate), n_feats, 3, padding=3 // 2),
        ]

        m_body = [
            ResBlock(
                dimension,
                n_feats,
                3,
                act=activation_fn[act](),  # , res_scale=res_scale
            )
            for _ in range(n_resblocks)
        ]

        m_body.append(conv_fn(n_feats, n_feats, 3, padding=3 // 2))

        m_tail = [
            conv_fn(n_feats, out_channels * prod(down_sample_rate), 3, padding=3 // 2),
            unshuffle,
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        return x


# class UnrollingRED(EDSR):
#     def __init__(self, dimension, n_resblocks, n_feats, in_channels=1, out_channels=1, , act='gelu'):
#         super().__init__()

#         self.config = config

#         arch_dict = architecture_dict(self.config)
#         self.net = arch_dict[self.config.method.network]()

#         self.adjkbnufft = AdjKbNufft(im_size=(IM_SIZE, IM_SIZE), grid_size=(GRID_SIZE, GRID_SIZE)).to(torch.float32)
#         self.kbnufft = KbNufft(im_size=(IM_SIZE, IM_SIZE), grid_size=(GRID_SIZE, GRID_SIZE)).to(torch.float32)

#         self.gamma = torch.nn.Parameter(torch.ones(1) * self.config.method.gamma_init)
#         #self.gamma=0.0001
#         self.tau = torch.nn.Parameter(torch.ones(1) * self.config.method.ured_tau_init)

#     def forward(self, ktraj, y, w, b1, is_trained=False):

#         b1 = b1_divided_by_rss(b1)
#         b1_input = b1

#         x0 = to_image_domain(self.adjkbnufft, y, b1, ktraj, w)

#         x = x0
#         x_hat = [x]
#         for _ in range(self.config.method.unrolling_steps):

#             dc = to_k_space(self.kbnufft, x, b1, ktraj)
#             dc = dc - y
#             dc = to_image_domain(self.adjkbnufft, dc, b1, ktraj, w)

#             prior = x.squeeze(2)
#             prior = prior.permute([0, 2, 1, 3, 4])
#             prior, minus_cof, divis_cof = batch_normalization_fn(prior)

#             prior = self.net(prior)

#             prior = batch_renormalization_fn(prior, minus_cof, divis_cof)
#             prior = prior.permute([0, 2, 1, 3, 4])
#             prior = prior.unsqueeze(2)

#             x = x - self.gamma * (dc + self.tau * (x - prior))
#             x_hat.append(x)

#         return x0, x_hat, b1_input, b1
