import re
from typing import Any

import lightning.pytorch as pl
import torch
import zarr
from einops import rearrange
from monai.inferers import PatchInferer, SlidingWindowSplitter
from monai.inferers.merger import AvgMerger
# import torchkbnufft as tkbn
# import torchopt
from dlboost.models import SD_RED
from dlboost.utils import to_png


# from torchmin import minimize

def bring_ref_phase_to_front(data, ref_idx, dim=1):
    # currently the shape of the kspace_data is (b, ph, ch, z, sp), randomly select a phase
    # and then select the corresponding kspace_data and kspace_traj as reference data,
    # then select the remaining kspace_data and kspace_traj as source data
    data_indices = list(range(data.shape[dim]))
    ref = data_indices.pop(ref_idx)
    data_indices = [ref] + data_indices
    return torch.index_select(data, dim, torch.tensor(data_indices).to(data.device))

def postprocessing(x, infer_overlap):
    b, ch, d, h, w = x.shape
    cut = d*infer_overlap//2
    x[:, :, 0:cut, ...] = 0
    x[:, :, -cut:, ...] = 0
    return 1/(1-infer_overlap)*x  # compensate for avg merger from monai

def nufft_adj_gpu(kspace_data, csm, kspace_traj, nufft_adj, inference_device, storage_device):
    image_init = nufft_adj(kspace_data.to(inference_device), kspace_traj.to(inference_device))
    result_reduced = torch.sum(image_init * csm.conj().to(inference_device), dim=1)
    return result_reduced.to(storage_device)

def forward_contrast(kspace_data, kspace_traj, recon_module, inferer,inference_device, storage_device):
    """
    kspace_data: [ph, ch, z, len]
    kspace_traj: [ph, 2, len]
    """
    with torch.no_grad():
        params = recon_module(kspace_data.to(inference_device), kspace_traj.to(inference_device))
        image, csm, mvf = params["image"], params["csm"], params["mvf"]
    return image, csm, mvf

# class ImageMerger()


class Recon(pl.LightningModule):
    def __init__(
        self,
        nufft_im_size=(320, 320),
        patch_size=(64, 64),
        iterations=5,
        ch_pad=42,
        lr = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.automatic_optimization = False
        self.recon_module = SD_RED(patch_size, nufft_im_size, ch_pad, iterations)
        self.lr = lr
        self.recon_loss_fn = torch.nn.L1Loss(reduction="none")
        infer_overlap = 0.5
        # self.inferer = PatchInferer(
        #     SlidingWindowSplitter((-1,patch_size[0],-1), overlap=(
        #         0, infer_overlap, 0), offset=(1, 0, 0)),
        #     ImageMerger(),
        #     batch_size=8,
        #     preprocessing=lambda x: x.to(self.device),
        #     postprocessing=lambda x: postprocessing(x, infer_overlap).to(torch.device("cpu")),
        #     value_dtype= torch.complex64)

    def training_step(self, batch, batch_idx):
        recon_opt = self.optimizers()
        kspace_data_odd, kspace_traj_odd = batch["kspace_data_z_fixed"], batch["kspace_traj_fixed"]
        kspace_data_even, kspace_traj_even = batch["kspace_data_z_moved"], batch["kspace_traj_moved"]

        ref_idx = torch.randint(0, 5, (1,))

        loss_a2b, params, image_list = self.n2n_step(kspace_data_odd, kspace_traj_odd, kspace_data_even, kspace_traj_even,ref_idx)
        recon_opt.zero_grad()

        # calculate the loss from even phase to odd phase
        loss_b2a, params, image_list = self.n2n_step(kspace_data_even, kspace_traj_even, kspace_data_odd, kspace_traj_odd,ref_idx)
        self.manual_backward(loss_b2a)
        self.log_dict({"recon/recon_loss": loss_b2a})
        recon_opt.step()

        image, csm, mvf = params["image"], params["csm"], params["mvf"]
        # if self.global_step % 1 == 0:
        for ch in [0, 3, 5]:
            to_png(self.trainer.default_root_dir+f'/csm_ch{ch}.png',
                   self.recon_module.forward_model.S._csm[0, 0, ch, 0, :, :])  # , vmin=0, vmax=2)
        for i in range(image.shape[0]):
            # to_png(self.trainer.default_root_dir+f'/image_recon{i}.png',
            #        image[i, 0, 0, :, :], vmin=0, vmax=5)
            for j,img in enumerate( image_list ):
                to_png(self.trainer.default_root_dir+f'/image_iter_{i}_{j}.png',
                       img[i, 0, 0, :, :] , vmin=0, vmax=5)
                # to_png(self.trainer.default_root_dir+f'/image_recon_ph{i}.png',
                #        image_recon_moved[i, 0, :, :])  # , vmin=0, vmax=2)
        

    def n2n_step(self, kspace_data_a, kspace_traj_a, kspace_data_b, kspace_traj_b, ref_idx):
        kspace_data_a_ = bring_ref_phase_to_front(kspace_data_a, ref_idx)
        kspace_traj_a_ = bring_ref_phase_to_front(kspace_traj_a, ref_idx)
        kspace_data_b_ = bring_ref_phase_to_front(kspace_data_b, ref_idx)
        kspace_traj_b_ = bring_ref_phase_to_front(kspace_traj_b, ref_idx)
        weight = torch.arange(
            1, kspace_data_b.shape[-1]//2+1, device=kspace_data_b.device)
        weight = torch.cat(
            [weight.flip(0), weight], dim=0)
        params, image_init = self.recon_module(kspace_data_a_, kspace_traj_a_, weight)
        image,csm,mvf = params["image"], params["csm"], params["mvf"]
        self.recon_module.forward_model.generate_forward_operators(mvf, csm, kspace_traj_b_)
        kspace_data_b_estimated = self.recon_module.forward_model(params)
        loss_not_reduced = \
            self.recon_loss_fn(torch.view_as_real(
                # kspace_data_b_estimated), torch.view_as_real(kspace_data_b_))
                weight*kspace_data_b_estimated), torch.view_as_real(kspace_data_b_*weight))
        loss = torch.mean(loss_not_reduced)
        return loss, params, image_init
        

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        image, csm, mvf = forward_contrast(
            batch["kspace_data_z"], batch["kspace_traj"],
            recon_module=self.recon_module,
            inferer = self.inferer,
            inference_device=self.device, storage_device=torch.device('cpu'))
        print(image.shape, csm.shape, mvf.shape)
        zarr.save(self.trainer.default_root_dir +
                    #   f'/epoch_{self.trainer.current_epoch}'+f'/image_init.zarr', image_init.abs().numpy(force=True))
                    f'/epoch_{self.trainer.current_epoch}'+'/image_recon.zarr', image[0, 0, :, 35:45].abs().numpy(force=True))
        zarr.save(self.trainer.default_root_dir +
                    #   f'/epoch_{self.trainer.current_epoch}'+f'/image_recon.zarr', image_recon.abs().numpy(force=True))
                    f'/epoch_{self.trainer.current_epoch}'+'/mvf.zarr', mvf[0, :, :, :, 35:45].abs().numpy(force=True))
        zarr.save(self.trainer.default_root_dir +
                    #   f'/epoch_{self.trainer.current_epoch}'+f'/csm.zarr', csm.abs().numpy(force=True))
                    f'/epoch_{self.trainer.current_epoch}'+'/csm.zarr', csm[0, :, :, :, 35:45].abs().numpy(force=True))
        print("Save image, csm and mvf to " + self.trainer.default_root_dir +
                f'/epoch_{self.trainer.current_epoch}')
        for ch in [0, 3, 5]:
            # to_png(self.trainer.default_root_dir+f'/epoch_{self.trainer.current_epoch}_'+f'/image_init_moved_ch{ch}.png',
            #        image_init_ch[0, ch, :, :])  # , vmin=0, vmax=2)
            to_png(self.trainer.default_root_dir+f'/epoch_{self.trainer.current_epoch}'+f'/csm_ch{ch}.png',
                    csm[0, :, ch, :, 35:45])  # , vmin=0, vmax=2)
        to_png(self.trainer.default_root_dir+f'/epoch_{self.trainer.current_epoch}'+'/image.png',
                image[0, 0, 40, :, :])  # , vmin=0, vmax=2)
            # to_png(self.trainer.default_root_dir+f'/epoch_{self.trainer.current_epoch}'+f'/image_recon_ph{i}.png',
            #         image_recon[i, 40, :, :])  # , vmin=0, vmax=2)

    def predict_step(self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0, device=torch.device("cuda"), ch_reduce_fn=torch.sum) -> Any:
        results = []
        for b in batch:
            # print(b["kspace_data_z"].shape, b["kspace_traj"].shape)
            image_recon, image_init, csm = forward_contrast(
                b["kspace_data_z_compensated"], b["kspace_traj"],
                b["kspace_data_z_cse"], b["kspace_traj_cse"],
                recon_module=self.recon_module, cse_forward=self.cse_forward,
                nufft_adj=self.nufft_adj_forward, inference_device=self.device, storage_device=torch.device('cpu'),
                inferer=self.inferer)
            # print(image_recon.shape, image_init.shape, csm.shape)
            results.append({
                'image_recon': image_recon,
                'image_init': image_init,
                'csm': csm,
                'id': b["id"]
                })
        return results

    def configure_optimizers(self):
        recon_optimizer = torch.optim.AdamW(
                self.parameters(),
            # [
            #     {"params": self.recon_module.parameters()},
            #     # {"params": self.cse_module.parameters()},
            # ], 
            lr=self.lr)
        return recon_optimizer

        
