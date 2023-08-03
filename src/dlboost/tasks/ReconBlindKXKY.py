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
        nufft_im_size=(320, 320),
        patch_size=(64, 64),
        recon_loss_fn=nn.MSELoss,
        recon_optimizer=optim.Adam,
        recon_lr = 1e-4,
        lambda_init = 2,
        eta = 1,
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

    def forward(self, x):
        return self.recon_module(x)
        
    def training_step(self, batch, batch_idx):
        recon_opt = self.optimizers()
        recon_opt.zero_grad()

        kspace_traj= batch['kspace_traj']
        kspace_data =  batch['kspace_data']
        kspace_data_compensated =  batch['kspace_data_compensated']
        w= batch["kspace_density_compensation"]
        # print(kspace_traj.shape, kspace_data.shape, kspace_data_compensated.shape, w.shape)
        lambda_ = self.lambda_init + 0.00028125 * self.global_step * kspace_data.shape[0]
        # generate a random discrete set for the kspace data
        seed = torch.rand(kspace_data.shape[-1], device=kspace_traj.device)
        masks = [torch.logical_and(seed >= 0+0.2*i, seed < 0.2*(i+1)) for i in range(5)]
        reverse_masks = [torch.logical_not(m) for m in masks]
        kspace_data_compensated_masked_list = [kspace_data_compensated[..., mask] for mask in reverse_masks]
        # kspace_data_masked_list = [kspace_data[..., mask] for mask in masks]
        kspace_traj_masked_list = [kspace_traj[..., mask] for mask in reverse_masks]
        # w_masked_list = [w[...,mask] for mask in masks]
        
        kspace_data_estimated_blind = torch.zeros_like(kspace_data)
        for k_compensated, k_traj, mask  in zip(kspace_data_compensated_masked_list, kspace_traj_masked_list, masks):
            # print(k_compensated.shape, k_traj.shape)
            image_init = self.nufft_adj_fn(k_compensated, k_traj)
            image_recon = self.forward(image_init)
            kspace_data_estimated = self.nufft_fn(image_recon, kspace_traj[..., mask])
            kspace_data_estimated_blind[..., mask] = kspace_data_estimated
        with torch.no_grad():
            image_init_unblind = self.nufft_adj_fn(kspace_data_compensated, kspace_traj)
            image_recon_unblind = self.forward(image_init_unblind)
            kspace_data_estimated_unblind = self.nufft_fn(image_recon_unblind, kspace_traj)
        loss_revisit = torch.mean( 
            (torch.view_as_real(kspace_data_estimated_blind) 
            # + lambda_ * torch.view_as_real(kspace_data_estimated_unblind)
            # - (lambda_+1) * torch.view_as_real(kspace_data))**2
             - torch.view_as_real(kspace_data))**2
            )
        loss_reg = torch.mean(
            self.eta * (torch.view_as_real(kspace_data_estimated_blind) 
                        + torch.view_as_real(kspace_data_estimated_unblind))**2)
        # breakpoint()
        self.manual_backward(loss_revisit+self.eta* loss_reg, retain_graph=True)
        self.log_dict({"recon/loss_revisit": loss_revisit})
        self.log_dict({"recon/loss_reg": loss_reg})
        self.log_dict({"recon/recon_loss": loss_reg+loss_revisit})

        if self.global_step%4==0:
            for i in range(image_init.shape[1]):
                to_png(self.trainer.default_root_dir+f'/image_init_ph{i}.png',
                       image_init[0, i, 0, :, :])#, vmin=0, vmax=2)
                to_png(self.trainer.default_root_dir+f'/image_recon_blind_ph{i}.png',
                       image_recon[0, i, 0, :, :])#, vmin=0, vmax=2)
                to_png(self.trainer.default_root_dir+f'/image_recon_unblind_ph{i}.png',
                       image_recon_unblind[0, i, 0, :, :])#, vmin=0, vmax=2)
        recon_opt.step()

    # def __training_step_recon(self, image_recon, kspace_traj, kspace_data, w):
    #     loss_image_space = self.recon_loss_fn(image_recon,image)
    #     kspace_data_estimated = self.nufft_fn(
    #         image_recon, kspace_traj)
    #     loss_not_reduced = \
    #         self.recon_loss_fn(torch.view_as_real(
    #             kspace_data_estimated), torch.view_as_real(kspace_data)) 
    #     loss_kspace = torch.mean(loss_not_reduced * w.unsqueeze(-1).expand(loss_not_reduced.shape)) 
    #     self.log("recon/loss_image_space", loss_image_space)
    #     self.log("recon/loss_kspace", loss_kspace)
    #     return loss_image_space + loss_kspace

    def validation_step(self, batch, batch_idx):
        for d in range(batch['cse'].shape[0]):
            image_recon, image_init = self.forward_contrast(batch["kspace_data_compensated"][d],
                                        batch["kspace_traj"][d], batch["cse"][d])
        return image_recon

    def forward_contrast(self, kspace_data_compensated, kspace_traj, cse):
        image_recon_list, image_init_list = [],[]
        cse_t = cse[0] 
        for kspace_data_t, kspace_traj_t in zip(kspace_data_compensated, kspace_traj):
            recon, init = self.forward_ch(
                kspace_data_compensated=kspace_data_t, kspace_traj=kspace_traj_t, cse=cse_t)
            image_recon_list.append(recon)
            image_init_list.append(init)
            break
        image_recon = torch.stack(image_recon_list)
        image_init = torch.stack(image_init_list)
        return image_recon, image_init
    
    def forward_ch(self, kspace_data_compensated, kspace_traj, cse):
        # image_recon_list, image_init_list = [],[]
        kspace_traj_ch = kspace_traj[0]
        ch = 0
        image_recon, image_init = 0, 0
        for kspace_data_ch,cse_ch in zip(kspace_data_compensated ,cse):
            recon, init = self.forward_step(
                kspace_data_compensated=kspace_data_ch.unsqueeze(0), kspace_traj=kspace_traj_ch.unsqueeze(0), cse=cse_ch.unsqueeze(0))
            image_recon = image_recon+recon
            image_init= image_init+init
            ch+=1
        zarr.save(f'tests/P2PKXKY_PhasCh_Even_Odd_Phase/image_recon_.zarr', image_recon.abs().numpy(force=True))
        zarr.save(f'tests/P2PKXKY_PhasCh_Even_Odd_Phase/image_init_.zarr', image_init.abs().numpy(force=True))
        return image_recon, image_init

    def forward_step(self, kspace_data_compensated, kspace_traj, cse, predictor=None):
        if predictor == None:
            predictor = self.recon_module
        image_init = self.nufft_adj_fn(kspace_data_compensated, kspace_traj)
        image_recon = sliding_window_inference(
            image_init, roi_size=self.patch_size,
            sw_batch_size=32, overlap=0, predictor=predictor) #, mode='gaussian')
        return image_recon*cse.conj(), image_init*cse.conj()

    def nufft_fn(self, image, omega):
        b, ph, c, l = omega.shape
        image_kx_ky_z = self.nufft_op( #torch.squeeze(image, dim=1)
                                   eo.rearrange(image, "b ph z x y -> (b ph) z x y"), 
                                   eo.rearrange(omega, "b ph c l -> (b ph) c l"), norm="ortho")
        image_kx_ky_z = eo.rearrange(
            image_kx_ky_z, "(b ph) z l -> b ph z l", b=b)
        # image_kx_ky_z.unsqueeze_(dim=1)
        return image_kx_ky_z

    def nufft_adj_fn(self, kdata, omega):
        b, ph, c, l = omega.shape
        image = self.nufft_adj(eo.rearrange(kdata, "b ph z l -> (b ph) z l"),
             eo.rearrange(omega, "b ph c l -> (b ph) c l"), norm="ortho")
        return eo.rearrange(image, "(b ph) z x y -> b ph z x y", b = b, ph = ph)

    def configure_optimizers(self):
        recon_optimizer = self.recon_optimizer(
            self.recon_module.parameters(), lr=self.recon_lr)
        return recon_optimizer
