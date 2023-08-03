
from email.mime import image
import random
import re
import napari
import zarr
from itertools import combinations_with_replacement, product, combinations
import lightning.pytorch as pl
import torch
from torch import nn, view_as_real
from torch.nn import functional as F
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
        self.regis_module = regis_module
        self.stn = SpatialTransformNetwork(STN_size)
        self.patch_size = patch_size
        self.nufft_im_size = nufft_im_size
        # self.fft_scale = torch.tensor(nufft_im_size).prod().sqrt()
        self.is_optimize_regis = is_optimize_regis
        self.lambda_ = lambda_
        self.loss_regis_mse_COEFF = loss_regis_mse_COEFF
        self.loss_recon_consensus_COEFF = loss_recon_consensus_COEFF
        self.recon_loss_fn = recon_loss_fn
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

        kspace_traj, kspace_data_compensated, kspace_data = batch['kspace_traj'], batch['kspace_data_compensated'], batch['kspace_data']
        # kspace_traj, kspace_data_compensated, kspace_data = batch['kspace_traj'].as_tensor(), batch['kspace_data_compensated'].as_tensor(), batch['kspace_data'].as_tensor()
        image_init = self.nufft_adj_fn(kspace_data_compensated, kspace_traj)
        
        for fixed_ph in range(image_init.shape[1]):
            image_init_fixed = image_init[:, fixed_ph, ...].unsqueeze(1)
            kspace_traj_fixed = kspace_traj[:, fixed_ph, ...]
            kspace_data_fixed = kspace_data[:, fixed_ph, ...].unsqueeze(1)
            # for i in range(image_init_1.shape[1]):

            # image_init_moved = image_init[:, moved_ph, ...].unsqueeze(1)
            # image_recon = self.forward(image_init_fixed)
            image_clean, noise_dependent, noise_independent = self.forward(image_init_fixed)
            if self.global_step %5 ==0 and fixed_ph == 0:
                to_png(self.trainer.default_root_dir+f'/image_init_1.png',
                    image_init_fixed[0, 0, 1, :, :], vmin=0, vmax=2)
                to_png(self.trainer.default_root_dir+f'/image_clean_1.png',
                    image_clean[0, 0, 1, :, :], vmin=0, vmax=2)
                to_png(self.trainer.default_root_dir+f'/noise_dep_1.png',
                    noise_dependent[0, 0, 1, :, :], vmin=0, vmax=2)
                to_png(self.trainer.default_root_dir+f'/noise_ind_1.png',
                    noise_independent[0, 0, 1, :, :], vmin=0, vmax=2)
            # image_clean, noise_dependent, noise_independent = torch.split(image_recon,1, dim=1)
            image_noisy = image_clean + noise_dependent + noise_independent
            loss_consistency = self.__training_step_recon(image_noisy, image_init_fixed)
            # loss_consistency = F.mse_loss(torch.view_as_real(image_noisy), torch.view_as_real( image_init_fixed ))
            
            # alpha = torch.rand(3, device=image_init.device)
            # alpha = alpha/alpha.sum()
            image_clean_2, noise_dependent_2, noise_independent_2 = self.forward(image_clean)
            image_clean_3, noise_dependent_3, noise_independent_3 = self.forward(image_clean+noise_dependent)
            image_clean_4, noise_dependent_4, noise_independent_4 = self.forward(noise_dependent+noise_independent)
            loss_identity = self.__training_step_recon(image_clean_2, image_clean)\
                + self.__training_step_recon(image_clean_3, image_clean)\
                + self.__training_step_recon(noise_dependent_3, noise_dependent)\
                + self.__training_step_recon(noise_dependent_4, noise_dependent)\
                + self.__training_step_recon(noise_independent_4, noise_independent)
                # F.mse_loss(torch.view_as_real(image_clean_2), torch.view_as_real(image_clean))\
                # + F.mse_loss(torch.view_as_real(image_clean_3), torch.view_as_real(image_clean))\
                # + F.mse_loss(torch.view_as_real(noise_dependent_3), torch.view_as_real(noise_dependent))\
                # + F.mse_loss(torch.view_as_real(noise_dependent_4), torch.view_as_real(noise_dependent))\
                # + F.mse_loss(torch.view_as_real(noise_independent_4), torch.view_as_real(noise_independent))
            loss_zero = \
              self.__training_step_recon(noise_dependent_2, torch.zeros_like(image_clean))\
            + self.__training_step_recon(noise_independent_2, torch.zeros_like(image_clean))\
            + self.__training_step_recon(noise_independent_3, torch.zeros_like(image_clean))\
            + self.__training_step_recon(image_clean_4, torch.zeros_like(image_clean))
            #   F.mse_loss(torch.view_as_real(noise_dependent_2), torch.zeros_like(torch.view_as_real(image_clean)))\
            # + F.mse_loss(torch.view_as_real(noise_independent_2), torch.zeros_like(torch.view_as_real(image_clean)))\
            # + F.mse_loss(torch.view_as_real(noise_independent_3), torch.zeros_like(torch.view_as_real(image_clean)))\
            # + F.mse_loss(torch.view_as_real(image_clean_4), torch.zeros_like(torch.view_as_real(image_clean)))
            total_loss = loss_consistency + loss_identity + loss_zero

            self.manual_backward(total_loss, retain_graph=True)
            self.log_dict({"recon/loss_consistency": loss_consistency}) 
            self.log_dict({"recon/loss_identity": loss_identity})
            self.log_dict({"recon/loss_zero": loss_zero})
            self.log_dict({"recon/total_loss": total_loss})
            # self.log_dict({"recon/recon_loss": recon_loss})
        recon_opt.step()
        self.untoggle_optimizer(recon_opt)
        return total_loss
    
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
        x_real = self.regis_module.spatial_transform(x.real, flow)
        x_imag = self.regis_module.spatial_transform(x.imag, flow)
        # return x
        return torch.complex(x_real, x_imag)

    def get_regis_losses(self, wrap, fixed, flow):
        regis_recon_loss = losses.ncc_loss(wrap, fixed)
        regis_grad_loss = losses.gradient_loss_3d(flow)
        regis_mse_loss = losses.mse_loss(wrap, fixed)
        return regis_recon_loss, regis_grad_loss, regis_mse_loss

    def __training_step_recon(self, image_recon, image_gt):
        # print(image_recon.shape, kspace_traj.shape)
        # kspace_data_estimated = self.nufft_fn(
        #     image_recon, kspace_traj)
        kspace_data_gt = torch.fft.fftshift(torch.fft.fft2(image_gt, dim=(-2, -1), norm = "ortho"), dim=(-2, -1))
        kspace_data_estimated = torch.fft.fftshift(torch.fft.fft2(image_recon, dim=(-2, -1), norm = "ortho"), dim=(-2, -1))
        loss = \
            self.recon_loss_fn(torch.view_as_real(kspace_data_estimated), torch.view_as_real(kspace_data_gt)) #+\
        return loss

    def validation_step(self, batch, batch_idx):
        # print(batch.shape)
        # batch = batch[0]
        for d in range(batch['cse'].shape[0]):
            image_recon, image_init = self.forward_contrast(batch["kspace_data_compensated"][d],
                                        batch["kspace_traj"][d], batch["cse"][d])
        return image_recon

    def forward_contrast(self, kspace_data_compensated, kspace_traj, cse):
        image_recon_list, image_init_list = [],[]
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
        for kspace_data_ch,cse_ch in zip(kspace_data_compensated ,cse):
            # print(kspace_traj_ch.shape, kspace_data_ch.shape, cse_ch.shape)
            recon, init = self.forward_step(
                kspace_data_compensated=kspace_data_ch.unsqueeze(0), kspace_traj=kspace_traj_ch.unsqueeze(0), cse=cse_ch.unsqueeze(0))
            image_recon = image_recon+recon
            image_init = image_init+init
            print(ch)
            ch+=1
        # image_recon = torch.sum(torch.cat(image_recon_list),dim=0)
        # image_init = torch.sum(torch.cat(image_init_list),dim=0)
        zarr.save(f'tests/DeCoLearn_KXKYZ_1blank_UNet/image_recon_.zarr', image_recon.abs().numpy(force=True))
        zarr.save(f'tests/DeCoLearn_KXKYZ_1blank_UNet/image_init_.zarr', image_init.abs().numpy(force=True))
        # viewer = napari.Viewer()
        # viewer.add_image(image_recon.abs().numpy(force=True))
        # viewer.add_image(image_init.abs().numpy(force=True))
        # napari.run()
        # pdb.set_trace()
        return image_recon, image_init

    def forward_step(self, kspace_data_compensated, kspace_traj,cse):
        image_init = self.nufft_adj_fn(kspace_data_compensated, kspace_traj)
        # image_recon = sliding_window_inference(
        #     image_init, roi_size=self.patch_size, 
        image_recon = sliding_window_inference(
            image_init, roi_size=self.patch_size, 
            sw_batch_size=8, predictor=self.recon_module, mode='gaussian')
        return image_recon*cse.conj(), image_init*cse.conj()
        # return self.recon_module(x)

    def nufft_fn(self, image, omega):
        b,  c, l = omega.shape
        image_kx_ky_z = self.nufft_op( #torch.squeeze(image, dim=1)
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
    # def nufft_fn(self, image, omega):
    #     # image is b c z x y (c=1) and type of image is complex64
    #     # shape_dict = eo.parse_shape(omega, "b _ sp len")
    #     b, _, sp, len = omega.shape
    #     image_kx_ky_z = self.nufft(torch.squeeze(image, dim=1), eo.rearrange(
    #         torch.view_as_real(omega), "b () sp len c -> b c (sp len)"), norm="ortho")
    #     image_kx_ky_z = eo.rearrange(
    #         image_kx_ky_z, "b z (sp len) -> b z sp len", sp=sp, len=len)
    #     image_kx_ky_z.unsqueeze_(dim=1)
    #     return image_kx_ky_z

    # def nufft_adj_fn(self, kdata, omega):
    #     # kdata is b ph z sp len and type of image is complex64
    #     # shape_dict = eo.parse_shape(omega, "b ph _ sp len")
    #     b, ph, _, sp, len = omega.shape
    #     image = self.nufft_adj(eo.rearrange(kdata, "b ph z sp len -> (b ph) z (sp len)"),
    #          eo.rearrange(torch.view_as_real(omega), "b ph () sp len c -> (b ph) c (sp len)"), norm="ortho")
    #     return eo.rearrange(image, "(b ph) z x y -> b ph z x y", b = b, ph = ph)

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
