import pdb
from email.mime import image
from typing import Any, Callable, Dict, Optional, Tuple, Union

import einops as eo
import lightning.pytorch as pl
import napari
import torch
import torchkbnufft as tkbn
import wandb
import zarr
from dlboost import losses
from dlboost.models.SpatialTransformNetwork import SpatialTransformNetwork
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
from torch import nn, optim
from torch.nn import functional as f
from torch.utils.hooks import RemovableHandle


def normalize(x, return_mean_std=False):
    mean = x.mean()
    std = x.std()
    if return_mean_std:
        return (x - mean) / std, mean, std
    else:
        return (x - mean) / std


def renormalize(x, mean, std):
    return x * std + mean


class Recon(pl.LightningModule):
    def __init__(
        self,
        recon_module: nn.Module,
        nufft_im_size=(320, 320),
        recon_loss_fn=nn.MSELoss,
        recon_optimizer=optim.Adam,
        recon_lr=1e-4,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["recon_module", "regis_module", "recon_loss_fn", "loss_fn"]
        )
        self.automatic_optimization = False
        self.recon_module = recon_module
        self.loss_recon_consensus_COEFF = 0.2
        self.recon_loss_fn = recon_loss_fn
        self.recon_lr = recon_lr
        self.recon_optimizer = recon_optimizer
        self.nufft_im_size = nufft_im_size
        self.nufft_op = tkbn.KbNufft(im_size=self.nufft_im_size)
        self.nufft_adj = tkbn.KbNufftAdjoint(im_size=self.nufft_im_size)

    def forward_recon(self, batch):
        image = batch["img"]

    def training_step(self, batch, batch_idx):
        recon_opt = self.optimizers()
        recon_opt.zero_grad()

        kspace_traj_fixed, kspace_traj_moved = (
            batch["kspace_traj_fixed"],
            batch["kspace_traj_moved"],
        )
        kspace_data_compensated_fixed, kspace_data_compensated_moved = (
            batch["kspace_data_compensated_fixed"],
            batch["kspace_data_compensated_moved"],
        )
        kspace_data_fixed, kspace_data_moved = (
            batch["kspace_data_fixed"],
            batch["kspace_data_moved"],
        )
        # breakpoint()

        image_fixed = (
            self.nufft_adj_fn(  # change nufft_fn to receive ktraj as complex number
                kspace_data_compensated_fixed, kspace_traj_fixed
            )
        )
        image_moved = self.nufft_adj_fn(
            kspace_data_compensated_moved, kspace_traj_moved
        )
        # print((image_fixed**2).mean(), image_fixed.std())
        # print((image_moved**2).mean(), image_moved.std())

        image_fixed_recon = real_2ch_as_complex(
            self.recon_module(complex_as_real_2ch(image_fixed))
        )
        image_moved_recon = real_2ch_as_complex(
            self.recon_module(complex_as_real_2ch(image_moved))
        )

        if self.global_step % 10 == 0:
            to_png(
                self.trainer.default_root_dir
                + "/image_input_fixed{self.global_step}.png",
                image_fixed[0, 0, 2, :, :],
                vmin=0,
                vmax=10,
            )
            to_png(
                self.trainer.default_root_dir
                + "/image_input_moved{self.global_step}.png",
                image_moved[0, 0, 2, :, :],
                vmin=0,
                vmax=10,
            )
            to_png(
                self.trainer.default_root_dir
                + "/image_recon_fixed_{self.global_step}.png",
                image_fixed_recon[0, 0, 2, :, :],
                vmin=0,
                vmax=10,
            )
            to_png(
                self.trainer.default_root_dir
                + "/image_recon_moved_{self.global_step}.png",
                image_moved_recon[0, 0, 2, :, :],
                vmin=0,
                vmax=10,
            )
        cross_loss = 0
        loss_m2f = self.__training_step_recon(
            image_recon=image_fixed_recon,
            # image_recon_mean=image_fixed_mean, image_recon_std=image_fixed_std,
            kspace_traj=kspace_traj_moved,
            kspace_data=kspace_data_moved,
            image=image_moved,
        )
        loss_f2m = self.__training_step_recon(
            image_recon=image_moved_recon,
            # image_recon_mean=image_moved_mean, image_recon_std=image_moved_std,
            kspace_traj=kspace_traj_fixed,
            kspace_data=kspace_data_fixed,
        )
        cross_loss = loss_m2f + loss_f2m
        self.log_dict({"recon/cross_loss": cross_loss})

        loss_fixed = self.__training_step_recon(
            image_recon=image_fixed_recon,
            # image_recon_mean=image_fixed_mean, image_recon_std=image_fixed_std,
            kspace_traj=kspace_traj_fixed,
            kspace_data=kspace_data_fixed,
        )
        loss_moved = self.__training_step_recon(
            image_recon=image_moved_recon,
            # image_recon_mean=image_moved_mean, image_recon_std=image_moved_std,
            kspace_traj=kspace_traj_moved,
            kspace_data=kspace_data_moved,
        )
        self_loss = loss_fixed + loss_moved
        self.log_dict({"recon/consensus_loss": self_loss})
        self.manual_backward(cross_loss + self.loss_recon_consensus_COEFF * self_loss)
        recon_opt.step()

        return cross_loss + self_loss

    def __training_step_recon(self, image_recon, kspace_traj, kspace_data, image=None):
        kspace_data_estimated = self.nufft(image_recon, kspace_traj)
        loss = self.recon_loss_fn(
            kspace_data_estimated.real, kspace_data.real
        ) + self.recon_loss_fn(kspace_data_estimated.imag, kspace_data.imag)
        return loss

    def validation_step(self, batch, batch_idx):
        cse = batch[0]["cse"]
        ch, d, ph, sp, lens = batch[0]["kspace_data_compensated"].shape
        kspace_data_compensated = eo.rearrange(
            batch[0]["kspace_data_compensated"], "ch d ph sp len -> (ch d) () ph sp len"
        )
        kspace_traj = eo.rearrange(
            batch[0]["kspace_traj"], "ch d ph sp len -> (ch d) ph sp len"
        )
        image_init_list = []
        image_recon_list = []
        for kd, traj in zip(
            torch.split(kspace_data_compensated, 8, dim=0),
            torch.split(kspace_traj, 8, dim=0),
        ):
            image_init = self.nufft_adj_fn(kd.cuda(), traj.cuda())
            image_init_list.append(image_init)
            image_recon_list.append(
                real_2ch_as_complex(
                    self.recon_module(complex_as_real_2ch(image_init))
                ).cpu()
            )
        image_init = eo.rearrange(
            torch.cat(image_init_list, dim=0).cpu(),
            "(ch d) () ph h w -> ch d ph h w",
            ch=ch,
            d=d,
        )
        image_init = (
            eo.reduce(
                image_init * cse.conj(), "ch d ph h w -> ph d h w", reduction="sum"
            )
            .abs()
            .numpy(force=True)
        )
        image_recon = eo.rearrange(
            torch.cat(image_recon_list, dim=0),
            "(ch d) () ph h w -> ch d ph h w",
            ch=ch,
            d=d,
        )
        image_recon = (
            eo.reduce(
                image_recon * cse.conj(), "ch d ph h w -> ph d h w", reduction="sum"
            )
            .abs()
            .numpy(force=True)
        )

        zarr.save("tests/image_recon_P2PKXKY.zarr", image_recon)
        zarr.save("tests/image_init_P2PKXKY.zarr", image_init)

        self.viewer = napari.Viewer()
        self.viewer.add_image(image_recon)
        self.viewer.add_image(image_init)
        napari.run()
        return image_recon

    def nufft(self, image, omega):
        # image is b c ph x y (c=1) and type of image is complex64
        shape_dict = eo.parse_shape(omega, "b ph sp len")
        image_kx_ky_z = self.nufft_op(
            eo.rearrange(image, "b ch ph x y -> (b ph) ch x y"),
            eo.rearrange(
                torch.view_as_real(omega), "b ph sp len c -> (b ph) c (sp len)"
            ),
            norm="ortho",
        )
        return eo.rearrange(
            image_kx_ky_z,
            "(b ph) ch (sp len) -> b ch ph sp len",
            b=shape_dict["b"],
            ph=shape_dict["ph"],
            sp=shape_dict["sp"],
            len=shape_dict["len"],
        )

    def nufft_adj_fn(self, kdata, omega):
        # image is b c ph x y (c=1) and type of image is complex64
        shape_dict = eo.parse_shape(omega, "b ph sp len")
        image = self.nufft_adj(
            eo.rearrange(kdata, "b ch ph sp len -> (b ph) ch (sp len)"),
            eo.rearrange(
                torch.view_as_real(omega), "b ph sp len c -> (b ph) c (sp len)"
            ),
            norm="ortho",
        )
        return eo.rearrange(
            image,
            "(b ph) ch x y -> b ch ph x y",
            b=shape_dict["b"],
            ph=shape_dict["ph"],
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if self.trainer.training:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        else:
            batch = batch
        return batch

    def configure_optimizers(self):
        recon_optimizer = self.recon_optimizer(
            self.recon_module.parameters(), lr=self.recon_lr
        )
        return recon_optimizer
