import pdb
import re
from email.mime import image
from typing import Any, Callable, Dict, Optional, Tuple, Union

import einops as eo
import lightning.pytorch as pl
import torch
import torchkbnufft as tkbn
from dlboost import losses
from dlboost.models.SpatialTransformNetwork import SpatialTransformNetwork
from dlboost.tasks.boilerplate import *
from dlboost.tasks.boilerplate_P2PCSE import *
from dlboost.utils import (
    complex_as_real_2ch,
    complex_as_real_ch,
    real_2ch_as_complex,
    to_png,
)
from matplotlib import pyplot as plt
from monai.inferers import sliding_window_inference
from monai.transforms import RandGridPatchd
from mrboost import computation as comp
from numpy import stack
from sympy import cse
from torch import nn, norm, optim
from torch.nn import functional as f


class Recon(pl.LightningModule):
    def __init__(
        self,
        recon_module: nn.Module,
        cse_module: nn.Module,
        nufft_im_size=(320, 320),
        patch_size=(64, 64),
        ch_pad=42,
        recon_loss_fn=nn.MSELoss,
        recon_optimizer=optim.Adam,
        recon_lr=1e-4,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "recon_module",
                "cse_module",
                "regis_module",
                "recon_loss_fn",
                "loss_fn",
            ]
        )
        self.automatic_optimization = False
        self.recon_module = recon_module
        self.cse_module = cse_module
        self.loss_recon_consensus_COEFF = 0.2
        self.recon_loss_fn = recon_loss_fn
        self.recon_lr = recon_lr
        self.recon_optimizer = recon_optimizer
        self.nufft_im_size = nufft_im_size
        self.nufft_op = tkbn.KbNufft(im_size=self.nufft_im_size)
        self.nufft_adj = tkbn.KbNufftAdjoint(im_size=self.nufft_im_size)
        self.teop_op = tkbn.ToepNufft()
        self.patch_size = patch_size
        self.ch_pad = ch_pad
        self.downsample = lambda x: interpolate(x, scale_factor=0.5, mode="bilinear")
        self.upsample = lambda x: interpolate(x, scale_factor=2, mode="bilinear")

    def forward(self, x):
        csm = self.cse(x)
        y = self.recon_module(csm.conj() * x)
        return y

    def training_step(self, batch, batch_idx):
        recon_opt = self.optimizers()
        recon_opt.zero_grad()
        batch = batch[0]
        # kspace data is in the shape of [ch, ph, sp]
        kspace_traj_fixed, kspace_traj_moved = (
            batch["kspace_traj"][0::2, ...],
            batch["kspace_traj"][1::2, ...],
        )
        kspace_data_fixed, kspace_data_moved = (
            batch["kspace_data_z"][0::2, ...],
            batch["kspace_data_z"][1::2, ...],
        )
        kspace_data_compensated_fixed, kspace_data_compensated_moved = (
            batch["kspace_data_z_compensated"][0::2, ...],
            batch["kspace_data_z_compensated"][1::2, ...],
        )

        image_init_fixed_ch = self.nufft_adj(
            kspace_data_compensated_fixed, kspace_traj_fixed, norm="ortho"
        )
        # shape is [ph, ch, h, w]
        csm_fixed = self.cse_forward(image_init_fixed_ch)
        image_init_fixed = torch.sum(image_init_fixed_ch * csm_fixed.conj(), dim=1)
        # shape is [ph, h, w]
        image_recon_fixed = self.recon_module(image_init_fixed.unsqueeze(0)).squeeze(0)
        loss_f2m = self.calculate_recon_loss(
            image_recon=image_recon_fixed.unsqueeze(1).expand_as(csm_fixed),
            csm=csm_fixed,
            kspace_traj=kspace_traj_moved,
            kspace_data=kspace_data_compensated_moved,
        )
        #  weight=weight_reverse_sample_density)
        self.manual_backward(loss_f2m, retain_graph=True)
        self.log_dict({"recon/recon_loss": loss_f2m})

        image_init_moved_ch = self.nufft_adj(
            kspace_data_compensated_moved, kspace_traj_moved, norm="ortho"
        )
        # shape is [ph, ch, h, w]
        csm_moved = self.cse_forward(image_init_moved_ch)
        image_init_moved = torch.sum(image_init_moved_ch * csm_moved.conj(), dim=1)
        # shape is [ph, h, w]
        image_recon_moved = self.recon_module(image_init_moved.unsqueeze(0)).squeeze(
            0
        )  # shape is [ph, h, w]
        loss_m2f = self.calculate_recon_loss(
            image_recon=image_recon_moved.unsqueeze(1).expand_as(csm_moved),
            csm=csm_moved,
            kspace_traj=kspace_traj_fixed,
            kspace_data=kspace_data_compensated_fixed,
        )
        #  weight=weight_reverse_sample_density)
        self.manual_backward(loss_m2f, retain_graph=True)

        if self.global_step % 4 == 0:
            for ch in [0, 3, 5]:
                to_png(
                    self.trainer.default_root_dir + f"/image_init_moved_ch{ch}.png",
                    image_init_moved_ch[0, ch, :, :],
                )  # , vmin=0, vmax=2)
                to_png(
                    self.trainer.default_root_dir + f"/csm_moved_ch{ch}.png",
                    csm_moved[0, ch, :, :],
                )  # , vmin=0, vmax=2)
            for i in range(image_init_moved.shape[0]):
                to_png(
                    self.trainer.default_root_dir + f"/image_init_ph{i}.png",
                    image_init_moved[i, :, :],
                )  # , vmin=0, vmax=2)
                to_png(
                    self.trainer.default_root_dir + f"/image_recon_ph{i}.png",
                    image_recon_moved[i, :, :],
                )  # , vmin=0, vmax=2)
        recon_opt.step()

    def cse_forward(self, image_init_ch):
        ph, ch = image_init_ch.shape[0:2]
        image_init_ch_lr = image_init_ch.clone()
        for i in range(3):
            image_init_ch_lr = self.downsample(image_init_ch_lr)
        # image_init_ch_lr = interpolate(image_init_ch, scale_factor=0.25, mode='bilinear')
        if ch < self.ch_pad:
            image_init_ch_lr = f.pad(
                image_init_ch_lr, (0, 0, 0, 0, 0, self.ch_pad - ch)
            )
        # print(image_init_ch_lr.shape)
        csm_lr = self.cse_module(
            rearrange(image_init_ch_lr, "ph ch h w -> () (ph ch) h w")
        )
        csm_lr = rearrange(csm_lr, "() (ph ch) h w -> ph ch h w", ph=ph)[:, :ch]
        csm_hr = csm_lr
        for i in range(3):
            csm_hr = self.upsample(csm_hr)
        # csm_hr = self.upsample(csm_lr)
        # csm_hr = interpolate(csm_lr, scale_factor=4, mode='bilinear')
        # devide csm by its root square of sum
        csm_hr_norm = csm_hr / torch.sqrt(
            torch.sum(torch.abs(csm_hr) ** 2, dim=1, keepdim=True)
        )
        return csm_hr_norm

    def calculate_recon_loss(
        self, image_recon, csm, kspace_traj, kspace_data, weight=None
    ):
        # kernel = tkbn.calc_toeplitz_kernel(kspace_traj, im_size=self.nufft_im_size)
        # image_HTH = self.teop_op(image_recon, kernel, smaps = csm, norm= 'ortho')
        kspace_recon = self.nufft_op(image_recon, kspace_traj, smaps=csm, norm="ortho")
        image_HT = self.nufft_adj(kspace_data, kspace_traj, norm="ortho")
        kspace_HHT = self.nufft_op(image_HT, kspace_traj, norm="ortho")

        loss_not_reduced = self.recon_loss_fn(
            torch.view_as_real(kspace_recon), torch.view_as_real(kspace_HHT)
        )
        loss = torch.mean(loss_not_reduced)
        return loss

    def validation_step(self, batch, batch_idx):
        validation_step(self, batch)

    def predict_step(
        self,
        batch: Any,
        batch_idx: int = 0,
        dataloader_idx: int = 0,
        device=torch.device("cuda"),
        ch_reduce_fn=torch.sum,
    ) -> Any:
        for b in batch:
            print(b["kspace_data_z"].shape, b["kspace_traj"].shape)
            image_recon, image_init, csm = forward_contrast(
                b["kspace_data_z"],
                b["kspace_traj"],
                recon_module=self.recon_module,
                cse_forward=self.cse_forward,
                nufft_adj=self.nufft_adj,
                inference_device=self.device,
                storage_device=torch.device("cpu"),
            )
            print(image_recon.shape, image_init.shape, csm.shape)
            zarr.save(
                self.trainer.default_root_dir
                + f"/epoch_{self.trainer.current_epoch}"
                + f"/image_init.zarr",
                image_init.abs().numpy(force=True),
            )
            zarr.save(
                self.trainer.default_root_dir
                + f"/epoch_{self.trainer.current_epoch}"
                + f"/image_recon.zarr",
                image_recon.abs().numpy(force=True),
            )
            zarr.save(
                self.trainer.default_root_dir
                + f"/epoch_{self.trainer.current_epoch}"
                + f"/csm.zarr",
                csm.abs().numpy(force=True),
            )

    def configure_optimizers(self):
        recon_optimizer = self.recon_optimizer(
            [
                {"params": self.recon_module.parameters()},
                {"params": self.cse_module.parameters(), "lr": self.recon_lr * 0.1},
            ],
            lr=self.recon_lr,
        )
        return recon_optimizer
