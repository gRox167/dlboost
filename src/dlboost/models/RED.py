from dataclasses import dataclass
from math import prod
from typing import Optional, Sequence, Tuple, Union

import torch
from einops.layers.torch import Rearrange
from sklearn.metrics import dcg_score
from torch import nn
from torch.nn import functional as F
from torchkbnufft import AdjKbNufft, KbNufft


class UnrollingRED(nn.Module):
    def __init__(
        self,
        data_consistency_module,
        regularization_module,
        iterations,
        gamma_init=1.0,
        tau_init=1.0,
    ):
        super().__init__()
        self.data_consistency_module = data_consistency_module
        self.regularization_module = regularization_module
        self.iterations = iterations
        self.gamma = torch.nn.Parameter(
            torch.ones(
                self.iterations,
            )
            * gamma_init
        )
        self.tau = torch.nn.Parameter(
            torch.ones(
                self.iterations,
            )
            * tau_init
        )

    def forward(self, x):
        # y = x.clone()
        for i in range(self.iterations):
            dc = self.data_consistency_module(x.image, x.data)
            reg = self.regularization_module(x)  # and update csm
            x.data = x.data - self.gamma * dc + self.tau * reg
        return x


class P2PCSEDataConsistencyModule(nn.Module):
    def __init__(self, kbnufft_op, adjkbnufft_op):
        super().__init__()
        self.kbnufft_op = kbnufft_op
        self.adjkbufft_op = adjkbnufft_op

    def kbnufft(self, x):
        return self.kbnufft_op(x.data, x.omega)

    def forward(self, x, y):
        # x_image = x.image.clone().detach()
        l = (
            1 / 2 * torch.norm(self.kbnufft(x) - y.data) ** 2
        )  # do i need to stop gradient here?
        gradient = torch.autograd.grad(l, x)  # , create_graph=True)
        return gradient


class P2PCSERegularizationModule(nn.Module):
    def __init__(self, recon_module, cse_module):
        super().__init__()
        self.recon_module = recon_module
        self.cse_module = cse_module

    def forward(self, x):
        x.csm = self.cse_module(x.csm)
        x_ch = self.recon_module(torch.sum(x.image * x.csm.conj(), dim=1))
        return x_ch * x.csm

        # b1 = b1_divided_by_rss(b1)
        # b1_input = b1

        # x0 = to_image_domain(self.adjkbnufft, y, b1, ktraj, w)

        # x = x0
        # x_hat = [x]
        # for _ in range(self.config.method.unrolling_steps):

        #     dc = to_k_space(self.kbnufft, x, b1, ktraj)
        #     dc = dc - y
        #     dc = to_image_domain(self.adjkbnufft, dc, b1, ktraj, w)

        #     prior = x.squeeze(2)
        #     prior = prior.permute([0, 2, 1, 3, 4])
        #     prior, minus_cof, divis_cof = batch_normalization_fn(prior)

        #     prior = self.net(prior)

        #     prior = batch_renormalization_fn(prior, minus_cof, divis_cof)
        #     prior = prior.permute([0, 2, 1, 3, 4])
        #     prior = prior.unsqueeze(2)

        #     x = x - self.gamma * (dc + self.tau * (x - prior))
        #     x_hat.append(x)

        # return x0, x_hat, b1_input, b1
