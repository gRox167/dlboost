from email.mime import image
from typing import Any, Callable, Dict, Optional, Tuple, Union
import napari
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


def normalize(x, return_mean_std=False):
    mean = x.mean()
    std = x.std()
    if return_mean_std:
        return (x-mean)/std, mean, std
    else:
        return (x-mean)/std


def renormalize(x, mean, std):
    return x*std+mean


class Recon(pl.LightningModule):
    def __init__(
        self,
        recon_module: nn.Module,
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
        self.recon_loss_fn = recon_loss_fn
        self.recon_lr = recon_lr
        self.recon_optimizer = recon_optimizer
    def forward_recon(self, batch):
        image = batch['img']
        
    def training_step(self, batch, batch_idx):
        recon_opt = self.optimizers()
        recon_opt.zero_grad()
        image_fixed = batch['img_fixed'].as_tensor()
        image_moved = batch['img_moved'].as_tensor()
        image_fixed_recon = self.recon_module(image_fixed)
        image_moved_recon = self.recon_module(image_moved)
        if self.global_step % 10 == 0:
            to_png(self.trainer.default_root_dir+'/image_input_fixed{self.global_step}.png',
                   image_fixed[0, 0, 2, :, :], vmin=0, vmax=4)
            to_png(self.trainer.default_root_dir+'/image_input_moved{self.global_step}.png',
                   image_moved[0, 0, 2, :, :], vmin=0, vmax=4)
            to_png(self.trainer.default_root_dir+'/image_recon_fixed_{self.global_step}.png',
                   image_fixed_recon[0, 0, 2, :, :], vmin=0, vmax=4)
            to_png(self.trainer.default_root_dir+'/image_recon_moved_{self.global_step}.png',
                   image_moved_recon[0, 0, 2, :, :], vmin=0, vmax=4)
        loss_m2f = \
            self.__training_step_recon(image_recon=image_fixed_recon,
                                       image=image_moved)
        loss_f2m = \
            self.__training_step_recon(image_recon=image_moved_recon,
                                       image = image_fixed)
        cross_loss = loss_m2f+loss_f2m
        self.log_dict({"recon/cross_loss": cross_loss})

        self.manual_backward(cross_loss)
        recon_opt.step()

        return cross_loss

    def __training_step_recon(self, image_recon, image=None):
        loss = self.recon_loss_fn(image_recon, image)
        return loss

    def validation_step(self, batch, batch_idx):
        self.viewer = napari.Viewer()
        image_init = batch[0]['img'].as_tensor()
        image_init = eo.rearrange(image_init, "c d ph h w -> d c ph h w")
        # image_moved = batch['img_moved'].as_tensor()
        # image_fixed_recon = torch.zeros_like(image_fixed,device='cpu')
        image_fixed_list = torch.split(image_init, 16, dim=0)
        image_recon_list = []
        for data_in in image_fixed_list:
            image_recon_list.append(self.recon_module(data_in.cuda()).cpu())
        image_recon = eo.rearrange(torch.cat(image_recon_list, dim = 0), "d () ph h w -> ph d h w").numpy(force=True)
        image_init = eo.rearrange(image_init, "d () ph h w -> ph d h w").numpy(force=True)
        example_phase = 0
        zarr.save('tests/image_recon_P2P.zarr', image_recon[example_phase])
        zarr.save('tests/image_init_P2P.zarr', image_init[example_phase])
      
        self.viewer.add_image(image_recon)
        self.viewer.add_image(image_init)
        napari.run()
        return image_recon

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if self.trainer.training:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        else:
            batch = batch
        return batch

    def configure_optimizers(self):
        recon_optimizer = self.recon_optimizer(
            self.recon_module.parameters(), lr=self.recon_lr)
        return recon_optimizer
