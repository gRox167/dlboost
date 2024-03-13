import random
from re import X
import napari
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

# from torch.utils.data import Dataloader
# import Dataloader from pytorch
from monai.data import DataLoader

from typing import Optional, Sequence

from monai.data import PatchIterd, Dataset, PatchDataset
from monai.transforms import RandGridPatchd
from monai.inferers import sliding_window_inference
import torchkbnufft as tkbn

from mrboost import computation as comp
from dlboost.models.SpatialTransformNetwork import SpatialTransformNetwork
from dlboost import losses
from dlboost.utils import (
    complex_as_real_2ch,
    real_2ch_as_complex,
    complex_as_real_ch,
    to_png,
)


def normalize(x, return_mean_std=False):
    mean = x.mean()
    std = x.std()
    if return_mean_std:
        return (x - mean) / std, mean, std
    else:
        return (x - mean) / std


def renormalize(x, mean, std):
    return x * std + mean


class ReconRegis(pl.LightningModule):
    def __init__(
        self,
        recon_module: nn.Module,
        regis_module: nn.Module,
        STN_size: int | Sequence = [64, 64, 64],
        patch_size=[64, 64, 64],
        patch_epoch: int = 4,
        batch_size: int = 2,
        nufft_im_size=(320, 320),
        is_optimize_regis: bool = False,
        lambda_: float = 6.0,
        loss_regis_mse_COEFF: float = 0.0,
        loss_recon_consensus_COEFF: float = 0.2,
        recon_loss_fn=nn.MSELoss,
        recon_optimizer=optim.Adam,
        recon_lr=1e-5,
        regis_optimizer=optim.Adam,
        regis_lr=1e-5,
        gradient_accumulation_steps: int = 4,
        unrolling_steps: int = 4,
        gamma=0.0001,
        tau=1,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["recon_module", "regis_module", "recon_loss_fn", "loss_fn"]
        )
        self.automatic_optimization = False
        self.recon_module = recon_module
        self.regis_module = regis_module
        self.stn = SpatialTransformNetwork(STN_size)
        self.patch_size = patch_size
        self.patch_epoch = patch_epoch
        self.batch_size = batch_size
        self.nufft_im_size = nufft_im_size
        self.recon_loss_fn = recon_loss_fn
        self.nufft_op = tkbn.KbNufft(im_size=self.nufft_im_size)
        self.nufft_adj = tkbn.KbNufftAdjoint(im_size=self.nufft_im_size)
        self.recon_lr = recon_lr
        self.regis_lr = regis_lr
        self.recon_optimizer = recon_optimizer
        self.regis_optimizer = regis_optimizer
        self.patch_sampler = RandGridPatchd(
            keys=["image_fixed", "image_moved"],  # ,'cse_fixed','cse_moved']
            patch_size=self.patch_size,
            overlap=0.25,
        )
        self.unrolling_steps = unrolling_steps
        self.gamma = gamma
        self.tau = tau
        self.viewer = napari.Viewer()

    def forward(self, x):
        return real_2ch_as_complex(self.recon_module(complex_as_real_2ch(x)), c=5)

    def forward_unrolled_red(self, batch):
        y = batch["kspace_data"]
        print(batch["kspace_density_compensation"].shape)
        omega = batch["kspace_traj"]
        kspace_density_compensation = batch["kspace_density_compensation"] / 3.5e-6
        image_init = self.nufft_adj_fn(y * kspace_density_compensation, omega)

        x = image_init
        for _ in range(self.unrolling_steps):
            dc = self.nufft(x, omega)
            dc = dc - y
            dc = self.nufft_adj_fn(dc, omega)

            prior = real_2ch_as_complex(
                sliding_window_inference(
                    complex_as_real_2ch(image_init),
                    roi_size=self.patch_size,
                    sw_batch_size=8,
                    predictor=self.recon_module,
                    mode="gaussian",
                )
            )

            x = x - self.gamma * (dc + self.tau * (x - prior))
            # W hat does this step want to do?
            # Refer "RED" and
            # "Deep Model-Based Architectures for Inverse Problems under Mismatched Priors" equation (5)
            # x_hat.append(x)
            # self.viewer.add_image(x.clone().abs().numpy(force=True), name=f"unrolled_{_}")
        image_recon = x * batch["cse"].conj()
        # napari.run()
        return image_recon, image_init * batch["cse"].conj()

    def training_step(self, batch, batch_idx):
        if self.is_optimize_regis:
            recon_opt, regis_opt = self.optimizers()
            # torch.nn.utils.clip_grad_value_(self.regis_module.parameters(), 1)
            regis_opt.zero_grad()
            recon_opt.zero_grad()
        else:
            recon_opt = self.optimizers()
            recon_opt.zero_grad()
        kspace_traj_fixed, kspace_traj_moved = (
            batch["kspace_traj_fixed"],
            batch["kspace_traj_moved"],
        )
        kspace_data_fixed, kspace_data_moved = (
            batch["kspace_data_fixed"],
            batch["kspace_data_moved"],
        )
        kspace_density_compensation_fixed = (
            batch["kspace_density_compensation_fixed"] / 3.5e-6
        )[:, None, None, :, :]
        # print(kspace_density_compensation_fixed.mean())
        kspace_density_compensation_moved = (
            batch["kspace_density_compensation_moved"] / 3.5e-6
        )[:, None, None, :, :]  # 1000 is a scale factor, let mean of it be 1
        # print(kspace_density_compensation_moved.mean())
        # kdc_fixed = (kspace_data_fixed*kspace_density_compensation_fixed).sum()/kspace_data_fixed.sum()
        # kdc_moved = (kspace_data_moved*kspace_density_compensation_moved).sum()/kspace_data_moved.sum()
        # pdb.set_trace()
        # nufft with norm="ortho" will scale each element in data by 1/sqrt(N) sqrt(320*320)
        image_fixed = self.nufft_adj_fn(
            # kspace_data_fixed, kspace_traj_fixed)
            kspace_data_fixed * kspace_density_compensation_fixed,
            kspace_traj_fixed,
        )
        image_moved = self.nufft_adj_fn(
            # kspace_data_moved, kspace_traj_moved)
            kspace_data_moved * kspace_density_compensation_moved,
            kspace_traj_moved,
        )
        # print((kspace_data_fixed*kspace_density_compensation_fixed).mean(),
        #       (kspace_data_fixed*kspace_density_compensation_fixed).std())
        # print((kspace_data_fixed).abs().mean(), (kspace_data_fixed).std())
        # print(image_fixed.abs().mean(), image_fixed.std())
        # AAT = self.nufft(image_fixed, kspace_traj_fixed)
        # print((AAT-kspace_data_fixed).abs().mean(),
        #       (AAT-kspace_data_fixed).std())

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
                vmax=4,
            )
            to_png(
                self.trainer.default_root_dir
                + "/image_input_moved{self.global_step}.png",
                image_moved[0, 0, 2, :, :],
                vmin=0,
                vmax=4,
            )
            to_png(
                self.trainer.default_root_dir
                + "/image_recon_fixed_{self.global_step}.png",
                image_fixed_recon[0, 0, 2, :, :],
                vmin=0,
                vmax=4,
            )
            to_png(
                self.trainer.default_root_dir
                + "/image_recon_moved_{self.global_step}.png",
                image_moved_recon[0, 0, 2, :, :],
                vmin=0,
                vmax=4,
            )
        regis_loss = 0
        if self.is_optimize_regis:
            wrap_m2f, wrap_f2m, regis_loss = self.training_step_regis(
                image_fixed_recon, image_moved_recon
            )
            self.manual_backward(regis_loss, retain_graph=True)
            regis_opt.step()
            local_patch_fixed = wrap_m2f
            local_patch_moved = wrap_f2m
        else:
            wrap_m2f, wrap_f2m = None, None
            local_patch_fixed = image_fixed_recon
            local_patch_moved = image_moved_recon
        # regis_opt.zero_grad()
        # self.manual_backward(regis_loss, retain_graph=True)
        # regis_opt.step()
        cross_loss = 0
        loss_m2f = self.__training_step_recon(
            image_recon=local_patch_fixed,
            # image_recon_mean=image_fixed_mean, image_recon_std=image_fixed_std,
            kspace_traj=kspace_traj_moved,
            kspace_data=kspace_data_moved,
            image=image_moved,
        )
        loss_f2m = self.__training_step_recon(
            image_recon=local_patch_moved,
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

    def training_step_regis(self, image_fixed_recon, image_moved_recon):
        # if self.is_optimize_regis:
        image_fixed_recon_abs = torch.sqrt(
            torch.sum(image_fixed_recon**2, dim=1, keepdim=True)
        )
        image_moved_recon_abs = torch.sqrt(
            torch.sum(image_moved_recon**2, dim=1, keepdim=True)
        )

        wrap_m2f_abs, flow_m2f = self.regis_module(
            image_moved_recon_abs, image_fixed_recon_abs
        )
        regis_recon_loss_m2f, regis_grad_loss_m2f, regis_mse_loss_m2f = (
            self.get_regis_losses(wrap_m2f_abs, image_fixed_recon_abs, flow_m2f)
        )

        wrap_f2m_abs, flow_f2m = self.regis_module(
            image_fixed_recon_abs, image_moved_recon_abs
        )
        regis_recon_loss_f2m, regis_grad_loss_f2m, regis_mse_loss_f2m = (
            self.get_regis_losses(wrap_f2m_abs, image_moved_recon_abs, flow_f2m)
        )

        self.log_dict(
            {
                "regis/ncc_loss_m2f": regis_grad_loss_m2f,
                "regis/grad_loss_m2f": regis_grad_loss_m2f,
                "regis/mse_loss_m2f": regis_mse_loss_m2f,
            }
        )
        regis_loss = regis_recon_loss_m2f + regis_recon_loss_f2m

        if self.lambda_ > 0:
            regis_loss += self.lambda_ * (regis_grad_loss_m2f + regis_grad_loss_f2m)

        if self.loss_regis_mse_COEFF > 0:
            regis_loss += self.loss_regis_mse_COEFF * (
                regis_mse_loss_m2f + regis_mse_loss_f2m
            )
        self.log_dict({"regis/total_loss": regis_loss})

        wrap_m2f = self.regis_complex(image_moved_recon, flow_m2f)
        wrap_f2m = self.regis_complex(image_fixed_recon, flow_f2m)
        return wrap_m2f, wrap_f2m, regis_loss

    def regis_complex(self, x, flow):
        x_list = torch.split(x, 1, dim=1)
        real = self.regis_module.spatial_transform(x_list[0], flow)
        imag = self.regis_module.spatial_transform(x_list[1], flow)
        return torch.cat([real, imag], dim=1)

    def get_regis_losses(self, wrap, fixed, flow):
        regis_recon_loss = losses.ncc_loss(wrap, fixed)
        regis_grad_loss = losses.gradient_loss_3d(flow)
        regis_mse_loss = losses.mse_loss(wrap, fixed)
        return regis_recon_loss, regis_grad_loss, regis_mse_loss

    def __training_step_recon(self, image_recon, kspace_traj, kspace_data, image=None):
        # print(kspace_data.mean(),kspace_data.std())
        kspace_data_estimated = self.nufft(image_recon, kspace_traj)
        # print(kspace_data_estimated.mean(),kspace_data_estimated.std())
        # kspace_data_image = self.nufft_adj_fn(kspace_data, kspace_traj)
        # kspace_data_estimated_image = self.nufft_adj_fn(kspace_data_estimated, kspace_traj)
        # to_png('tests/kspace_data_image_{self.global_step}.png',kspace_data_image[0,0,2,:,:],vmin=0,vmax=1)
        # to_png('tests/kspace_data_estimated_image_{self.global_step}.png',kspace_data_estimated_image[0,0,2,:,:])
        # print(image.mean(),image.var())
        # print((image-kspace_data_image).abs().sum())
        loss = self.recon_loss_fn(
            kspace_data_estimated.real, kspace_data.real
        ) + self.recon_loss_fn(kspace_data_estimated.imag, kspace_data.imag)
        return loss

    def validation_step(self, batch, batch_idx):
        image_recon_list, image_init_list = [], []
        # for k,v in batch.items():
        #     print(k,v.shape)
        for (
            kspace_data_ch,
            cse_ch,
            kspace_traj_ch,
            kspace_density_compensation_ch,
        ) in zip(
            batch["kspace_data"],
            batch["cse"],
            batch["kspace_traj"],
            batch["kspace_density_compensation"],
        ):
            recon, init = self.forward_unrolled_red(
                dict(
                    kspace_data=kspace_data_ch[None, ...],
                    kspace_traj=kspace_traj_ch[None, ...],
                    kspace_density_compensation=kspace_density_compensation_ch[
                        None, None, ...
                    ],
                    cse=cse_ch[None, ...],
                )
            )
            image_recon_list.append(recon.cpu())
            image_init_list.append(init.cpu())
        image_recon = (
            torch.sum(torch.cat(image_recon_list), dim=0).abs().numpy(force=True)
        )
        image_init = (
            torch.sum(torch.cat(image_init_list), dim=0).abs().numpy(force=True)
        )
        zarr.save("tests/image_recon_unrolledRED.zarr", image_recon)
        zarr.save("tests/image_init_unrolledRED.zarr", image_init)

        self.viewer.add_image(image_recon)
        self.viewer.add_image(image_init)
        napari.run()
        return image_recon

    def nufft(self, image, omega):
        # image is b c z x y (c=1) and type of image is complex64
        shape_dict = eo.parse_shape(omega, "b sp len")
        # pdb.set_trace()
        image_kx_ky_z = self.nufft_op(
            torch.squeeze(image, dim=1),
            eo.rearrange(torch.view_as_real(omega), "b sp len c -> b c (sp len)"),
            norm="ortho",
        )
        image_kx_ky_z = eo.rearrange(
            image_kx_ky_z,
            "b z (sp len) -> b z sp len",
            sp=shape_dict["sp"],
            len=shape_dict["len"],
        )
        image_kx_ky_z.unsqueeze_(dim=1)
        return image_kx_ky_z

    def nufft_adj_fn(self, kdata, omega):
        # image is b c z x y (c=1) and type of image is complex64
        shape_dict = eo.parse_shape(omega, "b sp len")
        # pdb.set_trace()
        kdata = eo.rearrange(kdata, "b 1 z sp len -> b z (sp len)")
        image = self.nufft_adj(
            kdata,
            eo.rearrange(torch.view_as_real(omega), "b sp len c -> b c (sp len)"),
            norm="ortho",
        )
        # image_kx_ky_kz = comp.fft_1D(image_kx_ky_z, dim=1)
        # image = eo.rearrange(
        #     image, "b z (sp len) -> b z sp len", sp=shape_dict["sp"], len=shape_dict["len"])
        image.unsqueeze_(dim=1)
        return image

    def configure_optimizers(self):
        if self.is_optimize_regis:
            regis_optimizer = self.regis_optimizer(
                self.regis_module.parameters(), lr=self.regis_lr
            )
            recon_optimizer = self.recon_optimizer(
                self.recon_module.parameters(), lr=self.recon_lr
            )
            return [regis_optimizer, recon_optimizer]
        else:
            recon_optimizer = self.recon_optimizer(
                self.recon_module.parameters(), lr=self.recon_lr
            )
            return recon_optimizer
