import random
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


def fill_whole_image_with_patch(image, patch, patch_loc, patch_shape):
    shape_dict = eo.parse_shape(image, 'b d h w')
    slices = [slice(0, shape_dict['b'])]+[slice(patch_loc[i], patch_loc[i] + patch_shape[i])
                             for i in range(len(patch_loc))]
    plist = torch.split(patch, 1, dim=0)
    p = plist[0]+1j*plist[1]
    image[slices] = p
    return image


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
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
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
        self.patch_epoch = patch_epoch
        self.batch_size = batch_size
        self.nufft_im_size = nufft_im_size
        self.is_optimize_regis = is_optimize_regis
        self.lambda_ = lambda_
        self.loss_regis_mse_COEFF = loss_regis_mse_COEFF
        self.loss_recon_consensus_COEFF = loss_recon_consensus_COEFF
        self.recon_loss_fn = recon_loss_fn
        self.nufft_op = tkbn.KbNufft(
            im_size=self.nufft_im_size).to(self.device)
        self.nufft_adj = tkbn.KbNufftAdjoint(
            im_size=self.nufft_im_size).to(self.device)
        self.recon_lr = recon_lr
        self.regis_lr = regis_lr
        self.recon_optimizer = recon_optimizer
        self.regis_optimizer = regis_optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.patch_sampler = RandGridPatchd(
            keys=['image_fixed',  'image_moved']  # ,'cse_fixed','cse_moved']
            , patch_size=self.patch_size, overlap=0.25)

    def forward(self, x):
        return self.recon_module(x)


    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch

    def training_step(self, batch, batch_idx):
        if self.is_optimize_regis:
            recon_opt, regis_opt = self.optimizers()
            # torch.nn.utils.clip_grad_value_(self.regis_module.parameters(), 1)
            regis_opt.zero_grad()
        else:
            recon_opt = self.optimizers()
            recon_opt.zero_grad()
        image_recon_fixed_cache, image_recon_moved_cache = batch['image_recon_fixed_cache'].to(
            self.device), batch['image_recon_moved_cache'].to(self.device)
        image_recon_fixed_cache, image_recon_fixed_cache_mean, image_recon_fixed_cache_std = normalize(
            image_recon_fixed_cache, return_mean_std=True)
        image_recon_moved_cache, image_recon_moved_cache_mean, image_recon_moved_cache_std = normalize(
            image_recon_moved_cache, return_mean_std=True)
        kspace_traj_fixed = batch['kspace_traj_fixed'].to(self.device)
        kspace_traj_moved = batch['kspace_traj_moved'].to(self.device)
        kspace_density_compensation_fixed = batch['kspace_density_compensation_fixed'].to(
            self.device)
        kspace_density_compensation_moved = batch['kspace_density_compensation_moved'].to(
            self.device)
        kspace_data_fixed = batch['kspace_data_fixed'].to(self.device)
        kspace_data_moved = batch['kspace_data_moved'].to(self.device)
        kspace_mask_fixed = batch['kspace_mask_fixed'].to(self.device)
        kspace_mask_moved = batch['kspace_mask_moved'].to(self.device)
        
        to_png('tests/image_input_fixed_cache_{self.global_step}.png',image_recon_fixed_cache[0,40,:,:])
        to_png('tests/image_input_moved_cache_{self.global_step}.png',image_recon_moved_cache[0,40,:,:])
        image_recon_fixed_cache, image_recon_moved_cache = self.update_whole_image(
            image_recon_fixed_cache, image_recon_moved_cache)
        to_png('tests/image_recon_fixed_cache_{self.global_step}.png',image_recon_fixed_cache[0,40,:,:])
        to_png('tests/image_recon_moved_cache_{self.global_step}.png',image_recon_moved_cache[0,40,:,:])
        # (self.nufft(batch["image_recon_fixed_cache"].cuda(),kspace_traj_fixed)*kspace_mask_fixed-kspace_data_fixed).abs().mean()
        patch_set = [(x, y) for x, y in zip(torch.split(batch["image_fixed"],
                                                        self.batch_size), torch.split(batch["image_moved"], self.batch_size))]
        random.shuffle(patch_set)
        patch_idx = 0
        for f, m in patch_set*self.patch_epoch:
            f = f.to(self.device)
            m = m.to(self.device)
            image_fixed_recon = self.recon_module(
                complex_as_real_2ch(f))
            image_moved_recon = self.recon_module(
                complex_as_real_2ch(m))
            regis_loss = 0
            if self.is_optimize_regis:
                wrap_m2f, wrap_f2m, regis_loss = self.training_step_regis(
                    image_fixed_recon, image_moved_recon)
                self.manual_backward(regis_loss,retain_graph=True)
                regis_opt.step()
            else:
                wrap_m2f, wrap_f2m = None, None
            # regis_opt.zero_grad()
            # self.manual_backward(regis_loss, retain_graph=True)
            # regis_opt.step()
            cross_loss = 0
            if self.is_optimize_regis:
                local_patch_fixed = wrap_m2f
                local_patch_moved = wrap_f2m
            else:
                local_patch_fixed = image_fixed_recon
                local_patch_moved = image_moved_recon
            loss_m2f, image_recon_fixed_cache = \
                self.__training_step_recon(local_patch=local_patch_fixed, image_recon_cache=image_recon_fixed_cache,
                                           image_recon_cache_mean=image_recon_fixed_cache_mean, image_recon_cache_std=image_recon_fixed_cache_std,
                                           kspace_traj=kspace_traj_moved, 
                                           kspace_data=kspace_data_moved*kspace_density_compensation_moved,
                                           kspace_mask=kspace_mask_moved)
            loss_f2m, image_recon_moved_cache = \
                self.__training_step_recon(local_patch=local_patch_moved, image_recon_cache=image_recon_moved_cache,
                                           image_recon_cache_mean=image_recon_moved_cache_mean, image_recon_cache_std=image_recon_moved_cache_std,
                                           kspace_traj=kspace_traj_fixed, 
                                           kspace_data=kspace_data_fixed*kspace_density_compensation_fixed,
                                           kspace_mask=kspace_mask_fixed)
            cross_loss = loss_m2f+loss_f2m
            self.log_dict({"recon/cross_loss": cross_loss})

            loss_fixed, image_recon_fixed_cache = \
                self.__training_step_recon(local_patch=image_fixed_recon, image_recon_cache=image_recon_fixed_cache,
                                           image_recon_cache_mean=image_recon_fixed_cache_mean, image_recon_cache_std=image_recon_fixed_cache_std,
                                           kspace_traj=kspace_traj_fixed, 
                                           kspace_data=kspace_data_fixed*kspace_density_compensation_fixed,
                                           kspace_mask=kspace_mask_fixed)
            loss_moved, image_recon_moved_cache = \
                self.__training_step_recon(local_patch=image_moved_recon, image_recon_cache=image_recon_moved_cache,
                                           image_recon_cache_mean=image_recon_moved_cache_mean, image_recon_cache_std=image_recon_moved_cache_std,
                                           kspace_traj=kspace_traj_moved, 
                                           kspace_data=kspace_data_moved*kspace_density_compensation_moved,
                                           kspace_mask=kspace_mask_moved)
            self_loss = loss_fixed+loss_moved
            self.log_dict({"recon/consensus_loss": self_loss})
            self.manual_backward(cross_loss+self_loss)
            recon_opt.step()
            image_recon_fixed_cache = image_recon_fixed_cache.detach()
            image_recon_moved_cache = image_recon_moved_cache.detach()
            patch_idx += 1

        return cross_loss+self_loss

    def update_whole_image(self, f, m):
        with torch.no_grad():
            # print(self.patch_size)
            f = complex_as_real_2ch(torch.unsqueeze(f, dim=0))
            f = sliding_window_inference(
                f, roi_size=(10,320,320), sw_batch_size=self.batch_size, predictor=self.recon_module)
            f = real_2ch_as_complex(f)
            m = complex_as_real_2ch(torch.unsqueeze(m, dim=0))
            m = sliding_window_inference(
                m, roi_size=(10,320,320), sw_batch_size=self.batch_size, predictor=self.recon_module)
            m = real_2ch_as_complex(m)
        image_recon_fixed_cache = torch.squeeze(f, dim=0).clone()
        image_recon_moved_cache = torch.squeeze(m, dim=0).clone()
        return image_recon_fixed_cache, image_recon_moved_cache

    def training_step_regis(self, image_fixed_recon, image_moved_recon):
        # if self.is_optimize_regis:
        image_fixed_recon_abs = torch.sqrt(
            torch.sum(image_fixed_recon ** 2, dim=1, keepdim=True))
        image_moved_recon_abs = torch.sqrt(
            torch.sum(image_moved_recon ** 2, dim=1, keepdim=True))

        wrap_m2f_abs, flow_m2f = self.regis_module(
            image_moved_recon_abs, image_fixed_recon_abs)
        regis_recon_loss_m2f, regis_grad_loss_m2f, regis_mse_loss_m2f = self.get_regis_losses(
            wrap_m2f_abs, image_fixed_recon_abs, flow_m2f)

        wrap_f2m_abs, flow_f2m = self.regis_module(
            image_fixed_recon_abs, image_moved_recon_abs)
        regis_recon_loss_f2m, regis_grad_loss_f2m, regis_mse_loss_f2m = self.get_regis_losses(
            wrap_f2m_abs, image_moved_recon_abs, flow_f2m)

        self.log_dict({"regis/ncc_loss_m2f": regis_grad_loss_m2f,
                      "regis/grad_loss_m2f": regis_grad_loss_m2f, "regis/mse_loss_m2f": regis_mse_loss_m2f})
        regis_loss = regis_recon_loss_m2f + regis_recon_loss_f2m

        if self.lambda_ > 0:
            regis_loss += self.lambda_ * \
                (regis_grad_loss_m2f + regis_grad_loss_f2m)

        if self.loss_regis_mse_COEFF > 0:
            regis_loss += self.loss_regis_mse_COEFF * \
                (regis_mse_loss_m2f + regis_mse_loss_f2m)
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

    """    
    def  training_step_recon(self, image_fixed_recon, image_moved_recon,
                            image_recon_fixed_cache, image_recon_fixed_cache_mean, image_recon_fixed_cache_std,
                            image_recon_moved_cache, image_recon_moved_cache_mean, image_recon_moved_cache_std,
                            wrap_m2f, wrap_f2m,
                            kspace_traj_fixed, kspace_traj_moved,
                            kspace_data_fixed, kspace_data_moved,
                            kspace_mask_fixed, kspace_mask_moved):

        for i in range(batch_size):
            patch_loc = [p[i] for p in patch_location_batched]
            patch_shape = [p[i] for p in patch_shape_batched]
            image_recon_fixed_cache = fill_whole_image_with_patch(
                image_recon_fixed_cache, image_fixed_recon[i], patch_loc, patch_shape)
            image_recon_moved_cache = fill_whole_image_with_patch(
                image_recon_moved_cache, image_moved_recon[i], patch_loc, patch_shape)
        # to_png('tests/image_recon_fixed_cache_filled_recon.png',image_recon_fixed_cache[0,10,:,:])
        # to_png('tests/image_recon_moved_cache_filled_recon.png',image_recon_moved_cache[0,10,:,:])
        kspace_data_fixed_estimated = self.nufft(
            renormalize(image_recon_fixed_cache, image_recon_fixed_cache_mean, image_recon_fixed_cache_std), kspace_traj_fixed)*kspace_mask_fixed
        kspace_data_moved_estimated = self.nufft(
            renormalize(image_recon_moved_cache, image_recon_moved_cache_mean, image_recon_moved_cache_std), kspace_traj_fixed)*kspace_mask_fixed
        self_consensus_loss = \
            self.recon_loss_fn(kspace_data_fixed_estimated.real, kspace_data_fixed.real) +\
            self.recon_loss_fn(kspace_data_fixed_estimated.imag, kspace_data_fixed.imag) +\
            self.recon_loss_fn(kspace_data_moved_estimated.real, kspace_data_moved.real) +\
            self.recon_loss_fn(
                kspace_data_moved_estimated.imag, kspace_data_moved.imag)
        self.log_dict({"recon/self_consensus_loss": self_consensus_loss})
        recon_loss += self.loss_recon_consensus_COEFF * self_consensus_loss
        return recon_loss, image_recon_fixed_cache, image_recon_moved_cache 
        """ 
    def __training_step_recon(self, local_patch,
                              image_recon_cache, image_recon_cache_mean, image_recon_cache_std,
                              kspace_traj, kspace_data, kspace_mask):
        patch_location_batched = local_patch.meta["location"]
        patch_shape_batched = local_patch.meta["spatial_shape"]
        batch_size = local_patch.shape[0]
        for i in range(batch_size):
            patch_loc = [p[i] for p in patch_location_batched]
            patch_shape = [p[i] for p in patch_shape_batched]
            image_recon_cache = fill_whole_image_with_patch(
                image_recon_cache, local_patch[i], patch_loc, patch_shape)
        # to_png('tests/image_recon_fixed_cache_filled_wrap.png',image_recon_fixed_cache[0,10,:,:])
        # to_png('tests/image_recon_moved_cache_filled_wrap.png',image_recon_moved_cache[0,10,:,:])
        # image_recon_fixed_cache = self.renormalize(image_recon_fixed_cache, )
        kspace_data_estimated = self.nufft(
            renormalize(image_recon_cache, image_recon_cache_mean, image_recon_cache_std), kspace_traj)*kspace_mask
        loss = \
            self.recon_loss_fn(kspace_data_estimated.real, kspace_data.real) +\
            self.recon_loss_fn(kspace_data_estimated.imag, kspace_data.imag)
        return loss, image_recon_cache

    def validation_step(self, batch, batch_idx):
        f = batch['image_fixed'].to(
            self.device)
        to_png('tests/val_image_input_10.png', f[0, 10, :, :])
        to_png('tests/val_image_input_40.png', f[0, 40, :, :])
        f = complex_as_real_2ch(torch.unsqueeze(f, dim=0))
        f = sliding_window_inference(f, roi_size=(
            20, 320, 320), sw_batch_size=1, predictor=self.recon_module, mode='gaussian')
        output = real_2ch_as_complex(f)
        pdb.set_trace()
        to_png('tests/val_image_recon_10.png', output[0, 0, 10, :, :])
        to_png('tests/val_image_recon_40.png', output[0, 0, 40, :, :])
        # img = comp.normalization(torch.squeeze(output,dim=0).abs()[40,:,:])
        # self.logger.experiment.log({"samples": wandb.Image(img)})
        # return {"pred": output}

    def nufft(self, image, omega):
        # omega = eo.rearrange(torch.view_as_real(omega),"sp len 2 -> 2 sp len")
        shape_dict = eo.parse_shape(omega, "sp len")
        image_kx_ky_z = self.nufft_op(image, eo.rearrange(
            torch.view_as_real(omega), "sp len c -> c (sp len)"), norm="ortho")
        image_kx_ky_kz = comp.fft_1D(image_kx_ky_z, dim=1)
        image_kx_ky_kz = eo.rearrange(
            image_kx_ky_kz, "b z (sp len) -> b z sp len", sp=shape_dict["sp"], len=shape_dict["len"])
        return image_kx_ky_kz

    def configure_optimizers(self):
        if self.is_optimize_regis:
            regis_optimizer = self.regis_optimizer(
                self.regis_module.parameters(), lr=self.regis_lr)
            recon_optimizer = self.recon_optimizer(
                self.recon_module.parameters(), lr=self.recon_lr)
            return [regis_optimizer, recon_optimizer]
        else:
            recon_optimizer = self.recon_optimizer(
                self.recon_module.parameters(), lr=self.recon_lr)
            return recon_optimizer
