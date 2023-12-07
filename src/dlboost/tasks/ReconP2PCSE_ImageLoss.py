from typing import Any
import lightning.pytorch as pl
from torchmetrics.image import TotalVariation
import torch
from torch import nn
from torch.nn import functional as f
from torch import optim

import torchkbnufft as tkbn

from dlboost.utils import to_png
from dlboost.tasks.boilerplate import *
from dlboost.tasks.boilerplate_P2PCSE import *


class Recon(P2PCSE):
    def __init__(
        self,
        recon_module: nn.Module,
        cse_module: nn.Module,
        nufft_im_size=(320, 320),
        patch_size=(64, 64),
        ch_pad=42,
        recon_loss_fn=nn.MSELoss,
        smooth_loss_coef = 0.1,
        recon_optimizer=optim.Adam,
        recon_lr=1e-4,
        **kwargs,
        ):
        super().__init__(recon_module, cse_module, nufft_im_size, patch_size, ch_pad, recon_loss_fn, smooth_loss_coef, recon_optimizer, recon_lr, **kwargs)

    def training_step(self, batch, batch_idx):
        recon_opt = self.optimizers()
        recon_opt.zero_grad()
        batch = batch[0]
        # kspace data is in the shape of [ch, ph, sp]
        kspace_traj_fixed, kspace_traj_moved = batch['kspace_traj'][
            0::2, ...], batch['kspace_traj'][1::2, ...]
        kspace_data_fixed, kspace_data_moved = batch['kspace_data_z'][
            0::2, ...], batch['kspace_data_z'][1::2, ...]
        kspace_data_compensated_fixed, kspace_data_compensated_moved = batch[
            'kspace_data_z_compensated'][
            0::2, ...],batch['kspace_data_z_compensated'][1::2, ...]


        csm_fixed = self.cse_forward(
            self.nufft_adj(kspace_data_fixed, kspace_traj_fixed, norm='ortho')
        )
        phase_diff_loss = torch.mean(torch.abs(
            csm_fixed[:1,...]- csm_fixed[1:, ...])**2)
        self.log_dict({"recon/phase_diff_loss": phase_diff_loss})
        # shape is [ph, ch, h, w]
        # csm_smooth_loss = self.smooth_loss_coef * gradient_loss(csm_fixed)
        # self.log_dict({"recon/csm_smooth_loss": csm_smooth_loss})

        image_init_fixed_ch = self.nufft_adj(
            kspace_data_compensated_fixed, kspace_traj_fixed, norm='ortho')
        image_init_fixed = torch.sum(
            image_init_fixed_ch * csm_fixed.conj(), dim=1)
        # shape is [ph, h, w]
        image_recon_fixed = self.recon_module(
            image_init_fixed.unsqueeze(0)).squeeze(0)
        loss_f2m = self.calculate_recon_loss(image_recon=image_recon_fixed.unsqueeze(1).expand_as(csm_fixed),
                                             csm=csm_fixed,
                                             kspace_traj=kspace_traj_moved,
                                             kspace_data=kspace_data_compensated_moved)
                                            #  weight=weight_reverse_sample_density)
        self.manual_backward(loss_f2m, retain_graph=True)
        # self.manual_backward(loss_f2m+csm_smooth_loss, retain_graph=True)
        self.log_dict({"recon/recon_loss": loss_f2m})

        csm_moved = self.cse_forward(
            self.nufft_adj(kspace_data_moved, kspace_traj_moved, norm='ortho')
        )
        phase_diff_loss = torch.mean(torch.abs(
            csm_moved[:1,...]- csm_moved[1:, ...])**2)
        # shape is [ph, ch, h, w]
        # csm_smooth_loss = self.smooth_loss_coef * gradient_loss(csm_moved)

        image_init_moved_ch = self.nufft_adj(
            kspace_data_compensated_moved, kspace_traj_moved, norm='ortho')
        image_init_moved = torch.sum(
            image_init_moved_ch * csm_moved.conj(), dim=1)
        # shape is [ph, h, w]
        image_recon_moved = self.recon_module(
            image_init_moved.unsqueeze(0)).squeeze(0)  # shape is [ph, h, w]
        loss_m2f = self.calculate_recon_loss(image_recon=image_recon_moved.unsqueeze(1).expand_as(csm_moved),
                                             csm=csm_moved,
                                             kspace_traj=kspace_traj_fixed,
                                             kspace_data=kspace_data_compensated_fixed)
                                            #  weight=weight_reverse_sample_density)
        self.manual_backward(loss_m2f , retain_graph=True)
        # self.manual_backward(loss_m2f + csm_smooth_loss, retain_graph=True)

        if self.global_step % 4 == 0:
            for i in range(image_init_moved.shape[0]):
                for ch in [0, 3, 5]:
                    to_png(self.trainer.default_root_dir+f'/image_init_moved_ch{ch}_ph{i}.png',
                           image_init_moved_ch[i, ch, :, :])  # , vmin=0, vmax=2)
                    to_png(self.trainer.default_root_dir+f'/csm_moved_ch{ch}_ph{i}.png',
                           csm_moved[i, ch, :, :])  # , vmin=0, vmax=2)
                to_png(self.trainer.default_root_dir+f'/image_init_ph{i}.png',
                       image_init_moved[i, :, :])  # , vmin=0, vmax=2)
                to_png(self.trainer.default_root_dir+f'/image_recon_ph{i}.png',
                       image_recon_moved[i, :, :])  # , vmin=0, vmax=2)
        recon_opt.step()

    def calculate_recon_loss(self, image_recon, csm, kspace_traj, kspace_data, weight=None):
        # kernel = tkbn.calc_toeplitz_kernel(kspace_traj, im_size=self.nufft_im_size)
        # image_HTH = self.teop_op(image_recon, kernel, smaps = csm, norm= 'ortho')

        image_HT = self.nufft_adj(kspace_data, kspace_traj, norm= 'ortho')
        # kspace_data_estimated = self.nufft_op(
        #     image_recon, kspace_traj, smaps=csm, norm='ortho')

        loss_not_reduced = \
            self.recon_loss_fn(torch.view_as_real(
                image_recon*csm), torch.view_as_real(image_HT))
        loss = torch.mean(loss_not_reduced)
        return loss

