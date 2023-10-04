from email.mime import image
from typing import Any, Callable, Dict, Optional, Tuple, Union
import napari
from numpy import diff, repeat
from torch.utils.hooks import RemovableHandle
import zarr
import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as f
from torch import optim
import einops as eo
import pdb
from matplotlib import pyplot as plt
import wandb

from monai.inferers import sliding_window_inference
import torchkbnufft as tkbn

from mrboost import computation as comp
from dlboost import losses
from dlboost.utils import complex_as_real_2ch, real_2ch_as_complex, complex_as_real_ch, to_png
from dlboost.tasks.boilerplate import *


class Recon(pl.LightningModule):
    def __init__(
        self,
        recon_module: nn.Module,
        nufft_im_size=(320, 320),
        patch_size=(64, 64),
        recon_loss_fn=nn.MSELoss,
        recon_optimizer=optim.Adam,
        recon_lr=1e-4,
        lambda_init=2,
        eta=1,
        weight_coef=1,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=['recon_module', 'regis_module', 'recon_loss_fn', 'loss_fn'])
        self.automatic_optimization = False
        self.recon_module = recon_module
        self.loss_recon_consensus_COEFF = 0.2
        self.lambda_init = lambda_init
        self.eta = eta
        self.recon_loss_fn = recon_loss_fn
        self.recon_lr = recon_lr
        self.recon_optimizer = recon_optimizer
        self.nufft_im_size = nufft_im_size
        self.nufft_op = tkbn.KbNufft(
            im_size=self.nufft_im_size)
        self.nufft_adj = tkbn.KbNufftAdjoint(
            im_size=self.nufft_im_size)
        self.patch_size = patch_size
        self.weight_kspace_ema = 0
        self.masker = ImageMasker(width=4)
        # torch.zeros(640, requires_grad=False)
        # self.

    def forward(self, x):
        return self.recon_module(x)

    def training_step(self, batch, batch_idx):
        recon_opt = self.optimizers()
        recon_opt.zero_grad()

        kspace_traj = batch['kspace_traj']
        kspace_data = batch['kspace_data']
        kspace_data_compensated = batch['kspace_data_compensated']
        image_init = nufft_adj_fn(kspace_data_compensated.flatten(-2), kspace_traj.flatten(-2), self.nufft_adj)
        image_masked_and_masks = [self.masker.mask(image_init, i) for i in range(16)]
        
        lambda_ = self.lambda_init + 0.00028125 * \
            self.global_step * kspace_data.shape[0]

        image_recon_blind= torch.zeros_like(image_init)
        for image_masked, mask, mask_inv in image_masked_and_masks:
            image_recon = self.forward(image_masked)
            image_recon_blind += image_recon * mask
        with torch.no_grad():
            image_recon_unblind =  self.forward(image_init)

        diff_revisit = (image_recon_blind + \
            lambda_ * image_recon_unblind - \
            (lambda_+1) * image_init)
        loss_revisit = torch.mean(diff_revisit*diff_revisit.conj())
        
        diff_reg = (image_recon_blind - image_recon_unblind)
        loss_reg = torch.mean(self.eta * (diff_reg* diff_reg.conj()))

        self.manual_backward(loss_revisit+loss_reg, retain_graph=True)
        self.log_dict({"recon/loss_revisit": loss_revisit})
        self.log_dict({"recon/loss_reg": loss_reg})
        self.log_dict({"recon/recon_loss": loss_reg+loss_revisit})
        if self.global_step % 4 == 0:
            for i in range(image_init.shape[1]):
                to_png(self.trainer.default_root_dir+f'/image_init_ph{i}.png',
                       image_init[0, i, 0, :, :])  # , vmin=0, vmax=2)
                to_png(self.trainer.default_root_dir+f'/image_masked_ph{i}.png',
                       image_masked[0, i, 0, :, :])  # , vmin=0, vmax=2)
                to_png(self.trainer.default_root_dir+f'/image_recon_blind_ph{i}.png',
                       image_recon[0, i, 0, :, :])  # , vmin=0, vmax=2)
                to_png(self.trainer.default_root_dir+f'/image_recon_unblind_ph{i}.png',
                       image_recon_unblind[0, i, 0, :, :])  # , vmin=0, vmax=2)
        recon_opt.step()

    def validation_step(self, batch, batch_idx):
        validation_step(self, batch, batch_idx)

    def configure_optimizers(self):
        recon_optimizer = self.recon_optimizer(
            self.recon_module.parameters(), lr=self.recon_lr)
        return recon_optimizer
