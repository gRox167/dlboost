from email.mime import image
from typing import Any, Callable, Dict, Optional, Tuple, Union
import napari
from numpy import repeat
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

from monai.transforms import RandGridPatchd
from monai.inferers import sliding_window_inference
import torchkbnufft as tkbn

from mrboost import computation as comp
from dlboost.models.SpatialTransformNetwork import SpatialTransformNetwork
from dlboost import losses
from dlboost.utils import complex_as_real_2ch, real_2ch_as_complex, complex_as_real_ch, to_png
from dlboost.tasks.boilerplate import *
from dlboost.tasks.boilerplate_P2PKXKY import *

class Recon(pl.LightningModule):
    def __init__(
        self,
        recon_module: nn.Module,
        nufft_im_size=(320, 320),
        patch_size=(64, 64),
        recon_loss_fn=nn.MSELoss,
        recon_optimizer=optim.Adam,
        recon_lr = 1e-4,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=['recon_module', 'regis_module', 'recon_loss_fn', 'loss_fn'])
        self.automatic_optimization = False
        self.recon_module = recon_module
        self.loss_recon_consensus_COEFF = 0.2
        self.recon_loss_fn = recon_loss_fn
        self.recon_lr = recon_lr
        self.recon_optimizer = recon_optimizer
        self.nufft_im_size = nufft_im_size
        self.nufft_op = tkbn.KbNufft(
            im_size=self.nufft_im_size)
        self.nufft_adj = tkbn.KbNufftAdjoint(
            im_size=self.nufft_im_size)
        self.patch_size = patch_size

    def forward(self, x):
        return self.recon_module(x)
        
    def training_step(self, batch, batch_idx):
        recon_opt = self.optimizers()
        recon_opt.zero_grad()

        kspace_traj_fixed, kspace_traj_moved = batch['kspace_traj'][:, 0::2, ...], batch['kspace_traj'][:, 1::2, ...]
        kspace_data_fixed, kspace_data_moved = batch['kspace_data'][:, 0::2, ...], batch['kspace_data'][:, 1::2, ...]
        kspace_data_compensated_fixed, kspace_data_compensated_moved = batch['kspace_data_compensated'][:, 0::2, ...], batch['kspace_data_compensated'][:, 1::2, ...]
        weight = torch.arange(1, kspace_data_fixed.shape[-1]//2+1, device=kspace_data_fixed.device)
        # weight = torch.ones(kspace_data_fixed.shape[-1]//2, device=kspace_data_fixed.device)
        weight_reverse_sample_density = torch.cat([weight.flip(0),weight], dim=0)
        image_init_fixed = nufft_adj_fn(
            kspace_data_compensated_fixed, kspace_traj_fixed, self.nufft_adj)

        image_recon_fixed = self.recon_module(image_init_fixed)
        loss_f2m = self.calculate_recon_loss(image_recon=image_recon_fixed,
                                    kspace_traj=kspace_traj_moved,
                                    kspace_data=kspace_data_moved,
                                    weight=weight_reverse_sample_density)
        self.manual_backward(loss_f2m, retain_graph=True)
        self.log_dict({"recon/recon_loss": loss_f2m})

        image_init_moved = nufft_adj_fn(
            kspace_data_compensated_moved, kspace_traj_moved, self.nufft_adj)
        image_recon_moved = self.recon_module(image_init_moved)
        loss_m2f = self.calculate_recon_loss(image_recon=image_recon_moved,
                                kspace_traj=kspace_traj_fixed,
                                kspace_data=kspace_data_fixed,
                                weight=weight_reverse_sample_density)
        self.manual_backward(loss_m2f, retain_graph=True)

        if self.global_step%4==0:
            for i in range(image_init_moved.shape[1]):
                to_png(self.trainer.default_root_dir+f'/image_init_ph{i}.png',
                       image_init_moved[0, i, 0, :, :])#, vmin=0, vmax=2)
                to_png(self.trainer.default_root_dir+f'/image_recon_ph{i}.png',
                       image_recon_moved[0, i, 0, :, :])#, vmin=0, vmax=2)
        recon_opt.step()

    def calculate_recon_loss(self, image_recon, kspace_traj, kspace_data, weight):
        kspace_data_estimated = nufft_fn(
            image_recon, kspace_traj, self.nufft_op)
        
        loss_not_reduced = \
            self.recon_loss_fn(torch.view_as_real(
                kspace_data_estimated*weight), torch.view_as_real(kspace_data*weight)) 
        loss = torch.mean(loss_not_reduced) 
        return loss

    def validation_step(self, batch, batch_idx):
        validation_step(self, batch, batch_idx)
        
    def predict_step(self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0, device = torch.device("cuda"), ch_reduce_fn = torch.sum) -> Any:
        return predict_step(batch, self.nufft_adj, self, self.patch_size, device = device, ch_reduce_fn = ch_reduce_fn)
        
    def configure_optimizers(self):
        recon_optimizer = self.recon_optimizer(
            self.recon_module.parameters(), lr=self.recon_lr)
        return recon_optimizer
