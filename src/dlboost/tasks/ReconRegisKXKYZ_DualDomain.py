
from email.mime import image
import random
import re
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
from monai.data import DataLoader

from typing import Optional, Sequence

from monai.data import PatchIterd, Dataset, PatchDataset
from monai.transforms import RandGridPatchd
from monai.inferers import sliding_window_inference
import torchkbnufft as tkbn

from mrboost import computation as comp
from dlboost.models.SpatialTransformNetwork import SpatialTransformNetwork
from dlboost import losses
from dlboost.utils import complex_as_real_2ch, real_2ch_as_complex, complex_as_real_ch, to_png, abs_real_2ch, fft2, ifft2


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
        krecon_module: nn.Module,
        regis_module: nn.Module,
        STN_size: int | Sequence = [64, 64, 64],
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
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=['recon_module', 'regis_module', 'recon_loss_fn', 'loss_fn'])
        self.automatic_optimization = False
        self.recon_module = recon_module
        self.krecon_module = krecon_module
        self.regis_module = regis_module
        self.stn = SpatialTransformNetwork(STN_size)
        self.nufft_im_size = nufft_im_size
        # self.fft_scale = torch.tensor(nufft_im_size).prod().sqrt()
        self.is_optimize_regis = is_optimize_regis
        self.lambda_ = lambda_
        self.loss_regis_mse_COEFF = loss_regis_mse_COEFF
        self.loss_recon_consensus_COEFF = loss_recon_consensus_COEFF
        self.recon_loss_fn = recon_loss_fn
        self.nufft = tkbn.KbNufft(
            im_size=self.nufft_im_size)
        self.nufft_adj = tkbn.KbNufftAdjoint(
            im_size=self.nufft_im_size)
        self.recon_lr = recon_lr
        self.regis_lr = regis_lr
        self.recon_optimizer = recon_optimizer
        self.regis_optimizer = regis_optimizer

    def forward(self, batch):
        pass
        # kspace_density_compensation = (
        #     batch['kspace_density_compensation']/3.5e-6)[:, None, None, :, :]

        # image_init = self.nufft_adj_fn(
        #     batch['kspace_data']*kspace_density_compensation, batch['kspace_traj'])
        # image_recon = real_2ch_as_complex(sliding_window_inference(
        #     complex_as_real_2ch(image_init), roi_size=self.patch_size, 
        #     sw_batch_size=8, predictor=self.recon_module, mode='gaussian'))*batch["cse"].conj()
        # # pdb.set_trace()

        # return image_recon, image_init*batch["cse"].conj()
        # return self.recon_module(x)
    
    # @torch.compile()
    def recon_forward(self, model, x):
        return real_2ch_as_complex(model(complex_as_real_2ch(x)), c=5)
        # x_ = self.recon_module(eo.rearrange(x, "b ph z x y c -> b (ph c) z x y"))
        # x_ = eo.rearrange(x_, "b (c cmplx) d h w -> b c d h w cmplx",c = x_.shape[1]//2, cmplx=2)
        # return x_

    def training_step(self, batch, batch_idx):
        if self.is_optimize_regis:
            regis_opt, recon_opt = self.optimizers()
        else:
            recon_opt = self.optimizers()

        kspace_traj, kspace_data_compensated, kspace_data = batch['kspace_traj'], batch['kspace_data_compensated'], batch['kspace_data']
        # kspace_traj, kspace_data_compensated, kspace_data = batch['kspace_traj'].as_tensor(), batch['kspace_data_compensated'].as_tensor(), batch['kspace_data'].as_tensor()
        image_init = self.nufft_adj_fn(kspace_data_compensated, kspace_traj)
        kgrid_init = fft2(image_init)
        to_png(self.trainer.default_root_dir+f'/kgrid_init.png',
                   kgrid_init[0, 1, 1, :, :], vmin=0, vmax=2)
        
        if self.is_optimize_regis and self.global_step%2==0:
            self.toggle_optimizer(regis_opt)
            regis_opt.zero_grad()
            image_recon = self.recon_forward(self.recon_module, image_init)
            for fixed_ph, moved_ph in combinations(range(image_recon.shape[1]), 2):
                image_recon_fixed = image_recon[:, fixed_ph, ...].unsqueeze(1)
                image_recon_moved = image_recon[:, moved_ph, ...].unsqueeze(1)
                wrap_m2f, wrap_f2m, regis_loss = self.training_step_regis(
                    image_recon_fixed, image_recon_moved)
                self.manual_backward(regis_loss, retain_graph=True)
            regis_opt.step()
            self.untoggle_optimizer(regis_opt)

        self.toggle_optimizer(recon_opt)
        recon_opt.zero_grad()   
        image_recon = self.recon_forward(self.recon_module, image_init)
        kgrid_recon = self.recon_forward(self.krecon_module, kgrid_init)
        image_from_kgrid_recon = ifft2(kgrid_recon)
        to_png(self.trainer.default_root_dir+f'/kgrid_recon.png',
                   kgrid_recon[0, 1, 1, :, :], vmin=0, vmax=2)

        # if self.global_step %  == 0:
        for i in range(image_init.shape[1]):
            to_png(self.trainer.default_root_dir+f'/image_init_ph{i}.png',
                   image_init[0, i, 1, :, :], vmin=0, vmax=2)
            to_png(self.trainer.default_root_dir+f'/image_recon_ph{i}.png',
                   image_recon[0, i, 1, :, :], vmin=0, vmax=2)
            to_png(self.trainer.default_root_dir+f'/image_from_kgrid_recon_ph{i}.png',
                   image_from_kgrid_recon[0, i, 1, :, :], vmin=0, vmax=2)
            
        for fixed_ph, moved_ph in combinations_with_replacement(range(image_recon.shape[1]), 2):
            if fixed_ph == moved_ph:
                image_recon_self = image_recon[:, fixed_ph, ...].unsqueeze(1)
                kspace_data_self = kspace_data[:, fixed_ph, ...].unsqueeze(1)
                kspace_traj_self = kspace_traj[:, fixed_ph, ...]
                image_from_kgrid_recon_self = image_from_kgrid_recon[:, fixed_ph, ...].unsqueeze(1)
                loss_kspace_image, loss_kspace_kgrid, dual_loss = self.__training_step_recon(image_recon=image_recon_self,
                                            image_from_kgrid_recon=image_from_kgrid_recon_self,
                                            kspace_traj=kspace_traj_self,
                                            kspace_data=kspace_data_self)
                print(f"loss_kspace_image: {loss_kspace_image}, loss_kspace_kgrid: {loss_kspace_kgrid}, dual_loss: {dual_loss}")
                kspace_loss = loss_kspace_image+ loss_kspace_kgrid
                recon_loss = 5*(kspace_loss + dual_loss)
            else:
                image_recon_fixed = image_recon[:, fixed_ph, ...].unsqueeze(1)
                kspace_data_fixed = kspace_data[:, fixed_ph, ...].unsqueeze(1)
                kspace_traj_fixed = kspace_traj[:, fixed_ph, ...]
                image_from_kgrid_recon_fixed = image_from_kgrid_recon[:, fixed_ph, ...].unsqueeze(1)
                image_recon_moved = image_recon[:, moved_ph, ...].unsqueeze(1)
                kspace_data_moved = kspace_data[:, moved_ph, ...].unsqueeze(1)
                kspace_traj_moved = kspace_traj[:, moved_ph, ...]
                image_from_kgrid_recon_moved = image_from_kgrid_recon[:, moved_ph, ...].unsqueeze(1)
                if self.is_optimize_regis:
                    image_recon_fixed_abs = image_recon_fixed.abs()
                    image_recon_moved_abs = image_recon_moved.abs()
                    wrap_m2f_abs, flow_m2f = self.regis_module(
                        image_recon_moved_abs, image_recon_fixed_abs)
                    wrap_f2m_abs, flow_f2m = self.regis_module(
                        image_recon_fixed_abs, image_recon_moved_abs)
                    wrap_m2f = self.regis_complex(image_recon_moved, flow_m2f)
                    wrap_f2m = self.regis_complex(image_recon_fixed, flow_f2m)
                    wrapk_m2f = self.regis_complex(image_from_kgrid_recon_moved, flow_m2f)
                    wrapk_f2m = self.regis_complex(image_from_kgrid_recon_fixed, flow_f2m)
                else:
                    wrap_m2f, wrap_f2m, wrapk_m2f, wrapk_f2m = image_recon_moved, image_recon_fixed, image_from_kgrid_recon_moved, image_from_kgrid_recon_fixed
                loss_m2f_kspace_image, loss_m2f_kspace_kgrid, loss_m2f_dual = self.__training_step_recon(image_recon=wrap_m2f,
                                        image_from_kgrid_recon=wrapk_m2f,
                                        kspace_traj=kspace_traj_fixed,
                                        kspace_data=kspace_data_fixed)
                loss_f2m_kspace_image, loss_f2m_kspace_kgrid, loss_f2m_dual = self.__training_step_recon(image_recon=wrap_f2m,
                                        image_from_kgrid_recon=wrapk_f2m,
                                            kspace_traj=kspace_traj_moved,
                                            kspace_data=kspace_data_moved)
                kspace_loss = loss_m2f_kspace_image+loss_f2m_kspace_image+ loss_m2f_kspace_kgrid+loss_f2m_kspace_kgrid
                print(f"loss_m2f_kspace_image: {loss_m2f_kspace_image}, loss_m2f_kspace_kgrid: {loss_m2f_kspace_kgrid}, loss_m2f_dual: {loss_m2f_dual}")
                dual_loss = loss_m2f_dual+loss_f2m_dual
                recon_loss = kspace_loss + dual_loss
                to_png(self.trainer.default_root_dir+f'/wrap_{moved_ph}2{fixed_ph}.png',
                        wrap_m2f[0, 0, 1, :, :], vmin=0, vmax=2)
                to_png(self.trainer.default_root_dir+f'/wrap_{fixed_ph}2{moved_ph}.png',
                        wrap_f2m[0, 0, 1, :, :], vmin=0, vmax=2)
            self.manual_backward(recon_loss, retain_graph=True)
            self.log_dict({"recon/recon_loss_kspace": kspace_loss, 
                           "recon/recon_loss_dual": dual_loss})
        recon_opt.step()
        self.untoggle_optimizer(recon_opt)
        return recon_loss
    
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

        if self.lambda_ > 0:
            regis_loss += self.lambda_ * \
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
        real = self.regis_module.spatial_transform(x.real, flow)
        imag = self.regis_module.spatial_transform(x.imag, flow)
        return torch.complex(real,imag)

    def get_regis_losses(self, wrap, fixed, flow):
        regis_recon_loss = losses.ncc_loss(wrap, fixed)
        regis_grad_loss = losses.gradient_loss_3d(flow)
        regis_mse_loss = losses.mse_loss(wrap, fixed)
        return regis_recon_loss, regis_grad_loss, regis_mse_loss

    def __training_step_recon(self, image_recon, image_from_kgrid_recon, kspace_traj, kspace_data):
        kspace_data_estimated_from_kgrid = self.nufft_fn(image_from_kgrid_recon, kspace_traj)
        kspace_data_estimated_from_image = self.nufft_fn(image_recon, kspace_traj)
        loss_kspace_image = \
            self.recon_loss_fn(torch.view_as_real(kspace_data_estimated_from_image), torch.view_as_real(kspace_data)) 
        loss_kspace_kgrid = \
            self.recon_loss_fn(torch.view_as_real(kspace_data_estimated_from_kgrid), torch.view_as_real(kspace_data))
        loss_dual_domain = self.recon_loss_fn(image_from_kgrid_recon.real, image_recon.real) +\
            self.recon_loss_fn(image_from_kgrid_recon.imag, image_recon.imag) 
        return loss_kspace_image, loss_kspace_kgrid, loss_dual_domain

    def validation_step(self, batch, batch_idx):
        image_recon_list, image_init_list = [],[]
        print(batch["kspace_data"][0].shape)
        print(len(batch["kspace_data"]))
        # pdb.set_trace()
        for kspace_data_ch,cse_ch in zip(batch["kspace_data"], batch["cse"]):
            recon, init = self.forward(
                dict(kspace_data=kspace_data_ch, kspace_traj=batch["kspace_traj"], kspace_density_compensation=batch["kspace_density_compensation" ], cse=cse_ch)
            )
            image_recon_list.append(recon.cpu())
            image_init_list.append(init.cpu())
        image_recon = torch.sum(torch.cat(image_recon_list),dim=0).abs().numpy(force=True)
        image_init = torch.sum(torch.cat(image_init_list),dim=0).abs().numpy(force=True)
        zarr.save('tests/image_recon.zarr', image_recon)
        zarr.save('tests/image_init.zarr', image_init)
        viewer = napari.Viewer()
        viewer.add_image(image_recon)
        viewer.add_image(image_init)
        napari.run()
        pdb.set_trace()
        # image_recon = self.forward(batch)
        return image_recon

        # img = comp.normalization(torch.squeeze(output,dim=0).abs()[40,:,:])
        # self.logger.experiment.log({"samples": wandb.Image(img)})
        # return {"pred": output}

    def nufft_fn(self, image, omega):
        b,  c, l = omega.shape
        image_kx_ky_z = self.nufft( #torch.squeeze(image, dim=1)
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
        return eo.rearrange(image, "(b ph) z x y -> b ph z x y", b = b, ph = ph)
    
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
