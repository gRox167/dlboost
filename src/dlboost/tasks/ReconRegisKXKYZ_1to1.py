
from email.mime import image
import random
import re
from shutil import move
from textwrap import wrap
import napari
import zarr
from itertools import combinations_with_replacement, product, combinations
import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as f
from torch import optim
import einops as eo
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
# allow_ops_in_compiled_graph()
import pdb
from matplotlib import pyplot as plt
import wandb
# from torch.utils.data import Dataloader
# import Dataloader from pytorch

from typing import Optional, Sequence

import monai
from monai.data import DataLoader
from monai.data import PatchIterd, Dataset, PatchDataset
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


class ReconRegis(pl.LightningModule):
    def __init__(
        self,
        recon_module: nn.Module,
        regis_module: nn.Module,
        STN_size: int | Sequence = [64, 64, 64],
        patch_size: int | Sequence = [64, 64, 64],
        nufft_im_size=(320, 320),
        is_optimize_regis: bool = False,
        loss_regis_grad_COEFF: float = 6.0,
        loss_regis_mse_COEFF: float = 0.0,
        loss_recon_consensus_COEFF: float = 0.2,
        recon_loss_fn=nn.MSELoss,
        recon_optimizer=optim.Adam,
        recon_lr=1e-5,
        regis_loss_fn=nn.MSELoss,
        regis_optimizer=optim.Adam,
        regis_lr=1e-5,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=['recon_module', 'regis_module', 'recon_loss_fn', 'loss_fn'])
        self.automatic_optimization = False
        self.recon_module = recon_module
        self.regis_module = regis_module
        # self.stn = SpatialTransformNetwork(STN_size)
        self.patch_size = patch_size
        self.nufft_im_size = nufft_im_size
        # self.fft_scale = torch.tensor(nufft_im_size).prod().sqrt()
        self.is_optimize_regis = is_optimize_regis
        self.loss_regis_grad_COEFF = loss_regis_grad_COEFF
        self.loss_regis_mse_COEFF = loss_regis_mse_COEFF
        self.loss_recon_consensus_COEFF = loss_recon_consensus_COEFF
        self.recon_loss_fn = recon_loss_fn
        self.regis_loss_fn= regis_loss_fn
        self.regis_grad_loss_fn = monai.losses.BendingEnergyLoss()
        self.nufft_op = tkbn.KbNufft(
            im_size=self.nufft_im_size)
        self.nufft_adj = tkbn.KbNufftAdjoint(
            im_size=self.nufft_im_size)
        self.recon_lr = recon_lr
        self.regis_lr = regis_lr
        self.recon_optimizer = recon_optimizer
        self.regis_optimizer = regis_optimizer

    # @torch.compile()

    def forward(self, x):
        # return real_2ch_as_complex(self.recon_module(complex_as_real_2ch(x)), c=5)
        return self.recon_module(x)

    def training_step(self, batch, batch_idx):
        if self.is_optimize_regis:
            regis_opt, recon_opt = self.optimizers()
        else:
            recon_opt = self.optimizers()

        kspace_traj, kspace_data_compensated, kspace_data = batch[
            'kspace_traj'], batch['kspace_data_compensated'], batch['kspace_data']
        # kspace_traj, kspace_data_compensated, kspace_data = batch['kspace_traj'].as_tensor(), batch['kspace_data_compensated'].as_tensor(), batch['kspace_data'].as_tensor()
        image_init = self.nufft_adj_fn(kspace_data_compensated, kspace_traj)

        for i in [0,4]:
            to_png(self.trainer.default_root_dir+f'/image_init_ph{i}.png',
                   image_init[0, i, 0, :, :], vmin=0, vmax=2)
        if self.is_optimize_regis: # and 
            self.toggle_optimizer(regis_opt)
            regis_opt.zero_grad()
            # image_recon = self.forward(image_init)
            for fixed_ph, moved_ph in combinations(range(image_init.shape[1]), 2):
                image_recon_fixed = self.forward(image_init[:, fixed_ph, ...].unsqueeze(1))
                kspace_data_fixed = kspace_data[:, fixed_ph, ...].unsqueeze(1)
                kspace_traj_fixed = kspace_traj[:, fixed_ph, ...]
                image_recon_moved = self.forward(image_init[:, moved_ph, ...].unsqueeze(1))
                kspace_data_moved = kspace_data[:, moved_ph, ...].unsqueeze(1)
                kspace_traj_moved = kspace_traj[:, moved_ph, ...]
                if image_recon_fixed.shape[2]==1:
                    image_recon_fixed = image_recon_fixed.squeeze(2)
                    image_recon_moved = image_recon_moved.squeeze(2)
                    wrap_m2f, wrap_f2m, regis_loss = self.training_step_regis(
                        image_recon_fixed, image_recon_moved)
                    wrap_m2f = wrap_m2f.unsqueeze(2)
                    wrap_f2m = wrap_f2m.unsqueeze(2)
                    image_recon_fixed = image_recon_fixed.unsqueeze(2)
                    image_recon_moved = image_recon_moved.unsqueeze(2)
                else:
                    wrap_m2f, wrap_f2m, regis_loss = self.training_step_regis(
                        image_recon_fixed, image_recon_moved)
                    
                if fixed_ph ==0 and moved_ph == 4:
                    to_png(self.trainer.default_root_dir+f'/wrap_{moved_ph}to0.png',
                       wrap_m2f[0, 0, 0, :, :], vmin=0, vmax=2)
                    to_png(self.trainer.default_root_dir+f'/wrap_0to{moved_ph}.png',
                       wrap_f2m[0, 0, 0, :, :], vmin=0, vmax=2)
                    to_png(self.trainer.default_root_dir+f'/image_recon_ph{moved_ph}.png',
                           image_recon_moved[0, 0, 0, :, :], vmin=0, vmax=2)
                    to_png(self.trainer.default_root_dir+f'/image_recon_ph{fixed_ph}.png',
                           image_recon_fixed[0, 0, 0, :, :], vmin=0, vmax=2)
                    to_png(self.trainer.default_root_dir+f'/diff_regis_ph{moved_ph}.png',
                           image_recon_moved[0, 0, 0, :, :]-wrap_m2f[0, 0, 0, :, :], vmin=0, vmax=2)
                    to_png(self.trainer.default_root_dir+f'/diff_regis_ph{fixed_ph}.png',
                           image_recon_fixed[0, 0, 0, :, :]-wrap_f2m[0, 0, 0, :, :], vmin=0, vmax=2)
                self.manual_backward(regis_loss, retain_graph=True)
            regis_opt.step()
            self.untoggle_optimizer(regis_opt)
        # if self.global_step % 2 == 0:
        self.toggle_optimizer(recon_opt)
        recon_opt.zero_grad()
        # image_recon = self.forward(image_init)
        for fixed_ph, moved_ph in combinations(range(image_init.shape[1]), 2):
            image_recon_fixed = self.forward(image_init[:, fixed_ph, ...].unsqueeze(1))
            kspace_data_fixed = kspace_data[:, fixed_ph, ...].unsqueeze(1)
            kspace_traj_fixed = kspace_traj[:, fixed_ph, ...]
            image_recon_moved = self.forward(image_init[:, moved_ph, ...].unsqueeze(1))
            kspace_data_moved = kspace_data[:, moved_ph, ...].unsqueeze(1)
            kspace_traj_moved = kspace_traj[:, moved_ph, ...]
            loss_f_self = self.__training_step_recon(image_recon=image_recon_fixed,
                                                  kspace_traj=kspace_traj_fixed,
                                                  kspace_data=kspace_data_fixed)
            loss_m_self = self.__training_step_recon(image_recon=image_recon_moved,
                                                  kspace_traj=kspace_traj_moved,
                                                  kspace_data=kspace_data_moved)
            recon_loss_self = loss_f_self + loss_m_self
            self.log_dict({"recon/recon_loss_self": recon_loss_self})
            self.manual_backward(recon_loss_self, retain_graph=True)
            if self.is_optimize_regis:
                if image_recon_fixed.shape[2]==1:
                    image_recon_fixed = image_recon_fixed.squeeze(2)
                    image_recon_moved = image_recon_moved.squeeze(2)
                    wrap_m2f_abs, flow_m2f = self.regis_module(
                        image_recon_moved.abs(), image_recon_fixed.abs())
                    wrap_f2m_abs, flow_f2m = self.regis_module(
                        image_recon_fixed.abs(), image_recon_moved.abs())
                    wrap_m2f = self.regis_complex(image_recon_moved, flow_m2f)
                    wrap_f2m = self.regis_complex(image_recon_fixed, flow_f2m)
                    wrap_m2f = wrap_m2f.unsqueeze(2)
                    wrap_f2m = wrap_f2m.unsqueeze(2)
                    image_recon_fixed = image_recon_fixed.unsqueeze(2)
                    image_recon_moved = image_recon_moved.unsqueeze(2)
                else:
                    wrap_m2f_abs, flow_m2f = self.regis_module(
                        image_recon_moved.abs(), image_recon_fixed.abs())
                    wrap_f2m_abs, flow_f2m = self.regis_module(
                        image_recon_fixed.abs(), image_recon_moved.abs())
                    wrap_m2f = self.regis_complex(image_recon_moved, flow_m2f)
                    wrap_f2m = self.regis_complex(image_recon_fixed, flow_f2m)
            else:
                wrap_m2f, wrap_f2m = image_recon_moved, image_recon_fixed
                if fixed_ph == 0 and moved_ph == 4:
                    to_png(self.trainer.default_root_dir+f'/image_recon_ph{moved_ph}.png',
                           image_recon_moved[0, 0, 0, :, :], vmin=0, vmax=2)
                    to_png(self.trainer.default_root_dir+f'/image_recon_ph{fixed_ph}.png',
                           image_recon_fixed[0, 0, 0, :, :], vmin=0, vmax=2)
            loss_m2f = self.__training_step_recon(image_recon=wrap_m2f,
                                                  kspace_traj=kspace_traj_fixed,
                                                  kspace_data=kspace_data_fixed)
            loss_f2m = self.__training_step_recon(image_recon=wrap_f2m,
                                                  kspace_traj=kspace_traj_moved,
                                                  kspace_data=kspace_data_moved)
            recon_loss_cross = loss_m2f + loss_f2m
            recon_loss = recon_loss_cross
            # if fixed_ph==moved_ph:
            #     recon_loss *= 5
            # to_png(self.trainer.default_root_dir+f'/wrap_{moved_ph}2{fixed_ph}.png',
            # wrap_m2f[0, 0, 1, :, :], vmin=0, vmax=2)
            # to_png(self.trainer.default_root_dir+f'/wrap_{fixed_ph}2{moved_ph}.png',
            #         wrap_f2m[0, 0, 1, :, :], vmin=0, vmax=2)
            self.log_dict({
                           "recon/recon_loss_cross": recon_loss_cross})
            self.manual_backward(recon_loss, retain_graph=True)
        recon_opt.step()
        self.untoggle_optimizer(recon_opt)
        # return recon_loss
        

    # @torch.compile()
    def training_step_regis(self, image_recon_fixed, image_recon_moved):
        image_recon_fixed_abs = image_recon_fixed.abs()
        image_recon_moved_abs = image_recon_moved.abs()

        wrap_m2f_abs, flow_m2f = self.regis_module(
            image_recon_moved_abs, image_recon_fixed_abs)
        regis_recon_loss_m2f, regis_grad_loss_m2f, regis_mse_loss_m2f = self.get_regis_losses(
            wrap_m2f_abs, image_recon_fixed_abs, flow_m2f)

        wrap_f2m_abs, flow_f2m = self.regis_module(
            image_recon_fixed_abs, image_recon_moved_abs)
        regis_recon_loss_f2m, regis_grad_loss_f2m, regis_mse_loss_f2m = self.get_regis_losses(
            wrap_f2m_abs, image_recon_moved_abs, flow_f2m)

        self.log_dict({"regis/ncc_loss_m2f": regis_recon_loss_m2f,
                      "regis/grad_loss_m2f": regis_grad_loss_m2f, "regis/mse_loss_m2f": regis_mse_loss_m2f})
        regis_loss = regis_recon_loss_m2f + regis_recon_loss_f2m

        if self.loss_regis_grad_COEFF > 0:
            regis_loss += self.loss_regis_grad_COEFF * \
                (regis_grad_loss_m2f + regis_grad_loss_f2m)

        if self.loss_regis_mse_COEFF > 0:
            regis_loss += self.loss_regis_mse_COEFF * \
                (regis_mse_loss_m2f + regis_mse_loss_f2m)
        self.log_dict({"regis/total_loss": regis_loss})

        wrap_m2f = self.regis_complex(image_recon_moved, flow_m2f)
        wrap_f2m = self.regis_complex(image_recon_fixed, flow_f2m)
        return wrap_m2f, wrap_f2m, regis_loss
        # return wrap_m2f, regis_loss

    def regis_complex(self, x, flow):
        # x_list = torch.split(x, 1, dim=1)
        x_real = self.regis_module.spatial_transform(x.real, flow)
        x_imag = self.regis_module.spatial_transform(x.imag, flow)
        # return x
        return torch.complex(x_real, x_imag)

    def get_regis_losses(self, wrap, fixed, flow):
        regis_recon_loss = self.regis_loss_fn(wrap, fixed)
        regis_grad_loss = self.regis_grad_loss_fn(flow)
        regis_mse_loss = losses.mse_loss(wrap, fixed)
        return regis_recon_loss, regis_grad_loss, regis_mse_loss

    def __training_step_recon(self, image_recon, kspace_traj, kspace_data, image=None):
        kspace_data_estimated = self.nufft_fn(
            image_recon, kspace_traj)
        loss = \
            self.recon_loss_fn(torch.view_as_real(
                kspace_data_estimated), torch.view_as_real(kspace_data))  # +\
        # self.recon_loss_fn(kspace_data_estimated.real, kspace_data.real) +\
        # self.recon_loss_fn(kspace_data_estimated.imag, kspace_data.imag)
        return loss

    def validation_step(self, batch, batch_idx):
        # print(batch.shape)
        # batch = batch[0]
        for d in range(batch['cse'].shape[0]):
            image_recon, image_init = self.forward_contrast(batch["kspace_data_compensated"][d],
                                                            batch["kspace_traj"][d], batch["cse"][d])
        return image_recon

    def forward_contrast(self, kspace_data_compensated, kspace_traj, cse):
        image_recon_list, image_init_list = [], []
        cse_t = cse[0]
        for kspace_data_t, kspace_traj_t in zip(kspace_data_compensated, kspace_traj):
            # print(kspace_traj_t.shape, kspace_data_t.shape, cse_t.shape)
            recon, init = self.forward_ch(
                kspace_data_compensated=kspace_data_t, kspace_traj=kspace_traj_t, cse=cse_t)
            image_recon_list.append(recon)
            image_init_list.append(init)
            break
        image_recon = torch.stack(image_recon_list)
        image_init = torch.stack(image_init_list)
        return image_recon, image_init

    def forward_ch(self, kspace_data_compensated, kspace_traj, cse):
        kspace_traj_ch = kspace_traj[0]
        ch = 0
        image_recon, image_init = 0, 0
        for kspace_data_ch, cse_ch in zip(kspace_data_compensated, cse):
            # print(kspace_traj_ch.shape, kspace_data_ch.shape, cse_ch.shape)
            recon, init = self.forward_step(
                kspace_data_compensated=kspace_data_ch.unsqueeze(0), kspace_traj=kspace_traj_ch.unsqueeze(0), cse=cse_ch.unsqueeze(0))
            image_recon = image_recon+recon
            image_init = image_init+init
            print(ch)
            ch += 1
        # image_recon = torch.sum(torch.cat(image_recon_list),dim=0)
        # image_init = torch.sum(torch.cat(image_init_list),dim=0)
        zarr.save(f'tests/DeCoLearn_KXKYZ_1blank_UNet/image_recon_.zarr',
                  image_recon.abs().numpy(force=True))
        zarr.save(f'tests/DeCoLearn_KXKYZ_1blank_UNet/image_init_.zarr',
                  image_init.abs().numpy(force=True))
        # viewer = napari.Viewer()
        # viewer.add_image(image_recon.abs().numpy(force=True))
        # viewer.add_image(image_init.abs().numpy(force=True))
        # napari.run()
        # pdb.set_trace()
        return image_recon, image_init

    def forward_step(self, kspace_data_compensated, kspace_traj, cse, predictor=None):
        if predictor == None:
            predictor = self.recon_module
        image_init = self.nufft_adj_fn(kspace_data_compensated, kspace_traj)
        image_recon = sliding_window_inference(
            image_init, roi_size=self.patch_size,
            sw_batch_size=8, predictor=predictor, mode='gaussian')
        return image_recon*cse.conj(), image_init*cse.conj()
        # return self.recon_module(x)

    def nufft_fn(self, image, omega):
        b,  c, l = omega.shape
        image_kx_ky_z = self.nufft_op(  # torch.squeeze(image, dim=1)
            eo.rearrange(image, "b () z x y -> b z x y"),
            eo.rearrange(omega, "b c l -> b c l"), norm="ortho")
        image_kx_ky_z = eo.rearrange(
            image_kx_ky_z, "b z l -> b () z l", b=b)
        # image_kx_ky_z.unsqueeze_(dim=1)
        return image_kx_ky_z

    def nufft_adj_fn(self, kdata, omega):
        b, ph, c, l = omega.shape
        image = self.nufft_adj(eo.rearrange(kdata, "b ph z l -> (b ph) z l"),
                               eo.rearrange(omega, "b ph c l -> (b ph) c l"), norm="ortho")
        return eo.rearrange(image, "(b ph) z x y -> b ph z x y", b=b, ph=ph)

    def configure_optimizers(self):
        if self.is_optimize_regis:
            regis_optimizer = self.regis_optimizer(
                self.regis_module.parameters(), lr=self.regis_lr)
            recon_optimizer = self.recon_optimizer(
                self.recon_module.parameters(), lr=self.recon_lr)
            #     {'params': self.regis_module.parameters(), 'lr': self.regis_lr},
            #     {'params':self.recon_module.parameters(), 'lr': self.recon_lr}], lr = self.recon_lr)
            return [regis_optimizer, recon_optimizer]
        else:
            recon_optimizer = self.recon_optimizer(
                self.recon_module.parameters(), lr=self.recon_lr)
            return recon_optimizer
