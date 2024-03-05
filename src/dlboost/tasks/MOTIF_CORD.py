from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
import zarr
from dlboost.models import ComplexUnet, DWUNet
from dlboost.models.MOTIF_CORD import MOTIF_CORD
from dlboost.utils import to_png
from dlboost.utils.tensor_utils import for_vmap, interpolate
from einops import rearrange
from monai.inferers import PatchInferer, SlidingWindowSplitter
from monai.inferers.merger import AvgMerger
from mrboost.computation import generate_nufft_op
from torch.nn import functional as F


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
    cut = d * infer_overlap // 2
    x[:, :, 0:cut, ...] = 0
    x[:, :, -cut:, ...] = 0
    return 1 / (1 - infer_overlap) * x  # compensate for avg merger from monai


def nufft_adj_gpu(
    kspace_data, csm, kspace_traj, nufft_adj, inference_device, storage_device
):
    image_init = nufft_adj(
        kspace_data.to(inference_device), kspace_traj.to(inference_device)
    )
    result_reduced = torch.sum(image_init * csm.conj().to(inference_device), dim=1)
    return result_reduced.to(storage_device)


# class ImageMerger()

nufft_op, nufft_adj_op = generate_nufft_op((320, 320))
nufft_cse = torch.vmap(nufft_op)  # image: b ch d h w; ktraj: b 2 len
nufft_adj_cse = torch.vmap(nufft_adj_op)  # kdata: b ch d len; ktraj: b 2 len
nufft = torch.vmap(torch.vmap(nufft_op))
# image: b ph ch d h w; ktraj: b ph 2 len
nufft_adj = torch.vmap(torch.vmap(nufft_adj_op))
# kdata: b ph ch d len


class Recon(pl.LightningModule):
    def __init__(
        self,
        nufft_im_size=(320, 320),
        patch_size=(64, 64),
        iterations=5,
        ch_pad=42,
        lr=1e-3,
        mvf_flag=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.automatic_optimization = False
        self.recon_module = MOTIF_CORD(patch_size, nufft_im_size, iterations)
        self.cse_module = ComplexUnet(
            in_channels=ch_pad,
            out_channels=ch_pad,
            spatial_dims=3,
            conv_net=DWUNet(
                in_channels=2 * ch_pad,
                out_channels=2 * ch_pad,
                spatial_dims=3,
                strides=((1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 1, 1)),
                kernel_sizes=((3, 7, 7), (3, 7, 7), (3, 7, 7), (3, 7, 7), (3, 7, 7)),
                features=(128, 256, 256, 256, 256),
            ),
        )
        self.cse_module.load_state_dict(torch.load("weights/cse_pretrain.pth"))
        self.lr = lr
        self.recon_loss_fn = torch.nn.L1Loss(reduction="none")
        self.nufft_im_size = nufft_im_size
        self.patch_size = patch_size
        self.ch_pad = ch_pad
        self.downsample = lambda x: interpolate(
            x, scale_factor=(1, 0.5, 0.5), mode="trilinear"
        )
        self.upsample = lambda x: interpolate(
            x, scale_factor=(1, 2, 2), mode="trilinear"
        )
        # self.inferer = PatchInferer(
        #     SlidingWindowSplitter(patch_size, overlap=(0.5, 0, 0), offset=(4, 0, 0)),
        #     batch_size=8,
        #     preprocessing=lambda x: x.to(self.device),
        #     postprocessing=lambda x: postprocessing(x).to(torch.device("cpu")),
        #     value_dtype=torch.complex64,
        # )
        # self.cse_forward_batched = torch.vmap(
        #     lambda image_init_ch: self.cse_forward(image_init_ch), 0, 0
        # )
        # self.cse_forward_batched = for_vmap(
        #     lambda image_init_ch: self.cse_forward(image_init_ch), 0, 0
        # )

    def training_step(self, batch, batch_idx):
        recon_opt = self.optimizers()
        recon_opt.zero_grad()
        # batch = batch[0]
        kspace_traj_odd, kspace_traj_even = (
            batch["kspace_traj_odd"],
            batch["kspace_traj_even"],
        )
        kspace_traj_cse_odd, kspace_traj_cse_even = (
            batch["kspace_traj_cse_odd"],
            batch["kspace_traj_cse_even"],
        )
        kspace_data_odd, kspace_data_even = (
            batch["kspace_data_odd"],
            batch["kspace_data_even"],
        )
        kspace_data_cse_odd, kspace_data_cse_even = (
            batch["kspace_data_cse_odd"],
            batch["kspace_data_cse_even"],
        )
        image_init_odd, image_init_even = (
            batch["P2PCSE_odd"][:, 2:3],
            batch["P2PCSE_even"][:, 2:3],
        )
        # print(kspace_data_odd)
        b, ph, ch, z, spl = kspace_data_odd.shape
        # mvf_odd, mvf_even = batch["mvf_odd"], batch["mvf_even"]
        mvf_odd, mvf_even = (
            torch.zeros(b, ph - 1, 3, z, 320, 320, device=kspace_data_odd.device),
            torch.zeros(b, ph - 1, 3, z, 320, 320, device=kspace_data_odd.device),
        )

        # kspace weighted loss
        sp, len = 15, 640
        weight = torch.arange(1, len // 2 + 1, device=kspace_data_odd.device)
        weight_reverse_sample_density = torch.cat(
            [weight.flip(0), weight], dim=0
        ).expand(sp, len)
        weight_reverse_sample_density = rearrange(
            weight_reverse_sample_density, "sp len -> (sp len)"
        )

        ref_idx = 2

        image_init_odd_ch = nufft_adj_cse(kspace_data_cse_odd, kspace_traj_cse_odd)
        csm_odd = self.cse_forward(image_init_odd_ch)
        csm_odd = csm_odd.unsqueeze(1).expand(-1, 5, -1, -1, -1, -1)
        loss_a2b, params, image_list = self.n2n_step(
            image_init_odd,
            kspace_data_odd,
            kspace_traj_odd,
            csm_odd,
            mvf_odd,
            kspace_data_even,
            kspace_traj_even,
            ref_idx,
            weight=weight_reverse_sample_density,
        )
        self.manual_backward(loss_a2b, retain_graph=True)

        image = params
        if self.global_step % 5 == 0:
            for ch in [0, 3, 5]:
                to_png(
                    self.trainer.default_root_dir + f"/csm_ch{ch}.png",
                    self.recon_module.forward_model.S._csm[0, 0, ch, 0, :, :],
                )  # , vmin=0, vmax=2)
            for i in range(image.shape[0]):
                # to_png(self.trainer.default_root_dir+f'/image_recon{i}.png',
                #        image[i, 0, 0, :, :], vmin=0, vmax=5)
                for j, img in enumerate(image_list):
                    to_png(
                        self.trainer.default_root_dir + f"/image_iter_{i}_{j}.png",
                        img[i, 0, 0, :, :],
                        vmin=0,
                        vmax=5,
                    )
                # to_png(self.trainer.default_root_dir+f'/image_recon_ph{i}.png',
                #        image_recon_moved[i, 0, :, :])  # , vmin=0, vmax=2)

        image_init_even_ch = nufft_adj_cse(kspace_data_cse_even, kspace_traj_cse_even)
        csm_even = self.cse_forward(image_init_even_ch)
        csm_even = csm_even.unsqueeze(1).expand(-1, 5, -1, -1, -1, -1)
        loss_b2a, params, image_list = self.n2n_step(
            image_init_even,
            kspace_data_even,
            kspace_traj_even,
            csm_even,
            mvf_even,
            kspace_data_odd,
            kspace_traj_odd,
            ref_idx,
            weight=weight_reverse_sample_density,
        )
        self.manual_backward(loss_b2a)
        self.log_dict({"recon/recon_loss": loss_b2a})
        recon_opt.step()

    def cse_forward(self, image_init_ch):
        b, ch = image_init_ch.shape[0:2]
        image_init_ch_lr = image_init_ch.clone()
        for i in range(3):
            image_init_ch_lr = self.downsample(image_init_ch_lr)
        if ch < self.ch_pad:
            image_init_ch_lr = F.pad(
                image_init_ch_lr, (0, 0, 0, 0, 0, 0, 0, self.ch_pad - ch)
            )
        csm_lr = self.cse_module(image_init_ch_lr)
        return csm_lr[:, :ch]
        # for i in range(3):
        #     csm_hr = self.upsample(csm_hr)
        # devide csm by its root square of sum
        # csm_hr_norm = csm_hr / torch.sqrt(
        #     torch.sum(torch.abs(csm_hr) ** 2, dim=1, keepdim=True)
        # )
        # return csm_hr

    def n2n_step(
        self,
        image_init_a,
        kspace_data_a,
        kspace_traj_a,
        csm_a,
        mvf_a,
        kspace_data_b,
        kspace_traj_b,
        ref_idx,
        weight=None,
    ):
        kspace_data_a_ = bring_ref_phase_to_front(kspace_data_a, ref_idx)
        kspace_traj_a_ = bring_ref_phase_to_front(kspace_traj_a, ref_idx)
        kspace_data_b_ = bring_ref_phase_to_front(kspace_data_b, ref_idx)
        kspace_traj_b_ = bring_ref_phase_to_front(kspace_traj_b, ref_idx)
        # image_init =
        params, image_list = self.recon_module(
            image_init_a, kspace_data_a_, kspace_traj_a_, mvf_a, csm_a
        )
        self.recon_module.forward_model.generate_forward_operators(
            mvf_a, csm_a, kspace_traj_b_
        )
        kspace_data_b_estimated = self.recon_module.forward_model(params)
        loss_not_reduced = self.recon_loss_fn(
            torch.view_as_real(
                # kspace_data_b_estimated), torch.view_as_real(kspace_data_b_))
                weight * kspace_data_b_estimated
            ),
            torch.view_as_real(kspace_data_b_ * weight),
        )
        loss = torch.mean(loss_not_reduced)
        return loss, params, image_list

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        torch.set_grad_enabled(True)
        batch = batch[0]
        t, ph, ch, z, spl = batch["kspace_data"].shape

        def plot_and_validation(
            image_init, kspace_data, kspace_traj, kspace_data_cse, kspace_traj_cse, mvf
        ):
            image_recon, csm = self.forward_contrast(
                image_init,
                kspace_data,
                kspace_traj,
                kspace_data_cse,
                kspace_traj_cse,
                mvf,
                storage_device=torch.device("cpu"),
            )
            print(image_init.shape, image_recon.shape, csm.shape)
            zarr.save(
                self.trainer.default_root_dir
                + f"/epoch_{self.trainer.current_epoch}"
                + "/image_init.zarr",
                image_init[:, :, :].abs().numpy(force=True),
            )  # b 1 d h w
            zarr.save(
                self.trainer.default_root_dir
                + f"/epoch_{self.trainer.current_epoch}"
                + "/image_recon.zarr",
                image_recon[:, :, :].abs().numpy(force=True),
            )  # b 1 d h w
            zarr.save(
                self.trainer.default_root_dir
                + f"/epoch_{self.trainer.current_epoch}"
                + "/csm.zarr",
                csm[:, :, :].abs().numpy(force=True),
            )  # b ch d h w
            print(
                "Save image_init, image_recon, csm to "
                + self.trainer.default_root_dir
                + f"/epoch_{self.trainer.current_epoch}"
            )
            for ch in [0, 3, 5]:
                to_png(
                    self.trainer.default_root_dir
                    + f"/epoch_{self.trainer.current_epoch}"
                    + f"/csm_moved_ch{ch}.png",
                    csm[0, ch, 4, :, :],
                )  # , vmin=0, vmax=2)
            for i in range(image_init.shape[2]):
                to_png(
                    self.trainer.default_root_dir
                    + f"/epoch_{self.trainer.current_epoch}"
                    + f"/image_init_ph{i}.png",
                    image_init[0, 0, i],
                )  # , vmin=0, vmax=2)
                to_png(
                    self.trainer.default_root_dir
                    + f"/epoch_{self.trainer.current_epoch}"
                    + f"/image_recon_ph{i}.png",
                    image_recon[0, 0, i],
                )  # , vmin=0, vmax=2)
            return image_recon

        # print(
        #     batch["P2PCSE"].shape,
        #     batch["kspace_data"].shape,
        #     batch["kspace_traj"].shape,
        #     batch["kspace_data_cse"].shape,
        #     batch["kspace_traj_cse"].shape,
        #     torch.zeros(t, ph - 1, 3, z, 320, 320).shape,
        # )
        return for_vmap(plot_and_validation, (0, 0, 0, 0, 0, 0), 0, 1)(
            batch["P2PCSE"][:, 2:3],
            batch["kspace_data"],
            batch["kspace_traj"],
            batch["kspace_data_cse"],
            batch["kspace_traj_cse"],
            torch.zeros(t, ph - 1, 3, z, 320, 320),
            # b["mvf"]
        )

    def predict_step(
        self,
        batch: Any,
        batch_idx: int = 0,
        dataloader_idx: int = 0,
        device=torch.device("cuda"),
        ch_reduce_fn=torch.sum,
    ) -> Any:
        torch.set_grad_enabled(True)
        results = []
        for b in batch:
            # print(b["kspace_data_z"].shape, b["kspace_traj"].shape)
            image_recon, image_init, csm = self.forward_contrast(
                b["kspace_data_z_compensated"],
                b["kspace_traj"],
                b["kspace_data_z_cse"],
                b["kspace_traj_cse"],
                inference_device=self.device,
                storage_device=torch.device("cpu"),
                inferer=self.inferer,
            )
            # print(image_recon.shape, image_init.shape, csm.shape)
            results.append(
                {
                    "image_recon": image_recon,
                    "image_init": image_init,
                    "csm": csm,
                    "id": b["id"],
                }
            )
        return results

    def configure_optimizers(self):
        recon_optimizer = torch.optim.AdamW(
            self.parameters(),
            # [
            #     {"params": self.recon_module.parameters()},
            #     # {"params": self.cse_module.parameters()},
            # ],
            lr=self.lr,
        )
        return recon_optimizer

    def forward_contrast(
        self,
        image_init,
        kspace_data,
        kspace_traj,
        kspace_data_cse,
        kspace_traj_cse,
        mvf,
        storage_device=torch.device("cpu"),
    ):
        """
        kspace_data: [b, ph, ch, z, len]
        kspace_traj: [b, ph, 2, len]
        """
        print(kspace_data_cse.shape, kspace_traj_cse.shape)
        csm = nufft_adj_cse(
            kspace_data_cse.to(self.device),
            kspace_traj_cse.to(self.device),
        )
        print(kspace_data.shape, kspace_traj.shape, csm.shape, mvf.shape)
        csm = self.cse_forward(csm)
        csm_mul_ph = csm.unsqueeze(1).expand(-1, 5, -1, -1, -1, -1)

        image_recon, image_recon_list = self.recon_module(
            image_init.to(self.device),
            kspace_data.to(self.device),
            kspace_traj.to(self.device),
            mvf.to(self.device),
            csm_mul_ph.to(self.device),
        )
        return image_recon.to(storage_device), csm

    def splitter(self, kspace_data, kspace_traj, csm, mvf, overlap=0.5):
        pad_size = (self.patch_size[0] - (self.patch_size[0] * overlap)) // 2
        _kd = F.pad(kspace_data, (0, 0, pad_size, pad_size))
        _kj = kspace_traj
        _csm = F.pad(csm, (0, 0, 0, 0, pad_size, pad_size))
        _mvf = F.pad(mvf, (0, 0, 0, 0, pad_size, pad_size))
        slices = [
            slice(i, i + self.patch_size[0])
            for i in range(0, _kd.shape[3], self.patch_size[0])
        ]
        return [
            (_kd[:, :, :, s], _kj, _csm[:, :, :, s], _mvf[:, :, :, s]) for s in slices
        ], slices

    def merger(self, results, slices):
        pad_size = (self.patch_size[0] - (self.patch_size[0] * 0.5)) // 2
        return torch.cat([r[:, :, pad_size:-pad_size] for r in results], dim=3)

    def on_validation_model_eval(self) -> None:
        super().on_validation_model_eval()
        torch.set_grad_enabled(True)
