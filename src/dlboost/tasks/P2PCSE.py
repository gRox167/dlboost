from pathlib import Path
from typing import Any

import dask.array as da
import lightning as L
import torch
import torchkbnufft as tkbn
import xarray as xr
import zarr
from dlboost.models import ComplexUnet, DWUNet
from dlboost.utils import to_png
from dlboost.utils.io_utils import multi_processing_save_data
from dlboost.utils.patch_utils import cutoff_filter, infer, split_tensor
from dlboost.utils.tensor_utils import for_vmap, interpolate
from einops import rearrange  # , reduce, repeat
from jaxtyping import Shaped
from mrboost.computation import nufft_2d, nufft_adj_2d
from plum import activate_autoreload

activate_autoreload()
from dlboost.utils.type_utils import (
    ComplexImage2D,
    ComplexImage3D,
    KspaceData,
    KspaceTraj,
)
from plum import dispatch, overload
from torch import P, optim
from torch.nn import functional as f


class Recon(L.LightningModule):
    def __init__(
        self,
        nufft_im_size=(320, 320),
        patch_size=(5, 64, 64),
        ch_pad=42,
        lr=1e-4,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["recon_module", "cse_module", "regis_module", "recon_loss_fn"]
        )
        self.recon_module = ComplexUnet(
            in_channels=5,
            out_channels=5,
            spatial_dims=3,
            conv_net=DWUNet(
                in_channels=10,
                out_channels=10,
                spatial_dims=3,
                strides=((2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2)),
                kernel_sizes=((3, 7, 7), (3, 7, 7), (3, 7, 7), (3, 7, 7), (3, 7, 7)),
            ),
        )
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
        self.automatic_optimization = False
        self.loss_recon_consensus_COEFF = 0.2
        self.recon_loss_fn = torch.nn.L1Loss(reduction="none")
        self.recon_lr = lr
        self.recon_optimizer = optim.AdamW
        self.nufft_im_size = nufft_im_size
        # self.nufft_op = tkbn.KbNufft(im_size=self.nufft_im_size)
        # self.nufft_adj = tkbn.KbNufftAdjoint(im_size=self.nufft_im_size)
        # self.teop_op = tkbn.ToepNufft()
        self.patch_size = patch_size
        self.ch_pad = ch_pad
        self.downsample = lambda x: interpolate(
            x, scale_factor=(1, 0.5, 0.5), mode="trilinear"
        )
        self.upsample = lambda x: interpolate(
            x, scale_factor=(1, 2, 2), mode="trilinear"
        )
        # self.inferer = PatchInferer(
        #     SlidingWindowSplitter(patch_size, overlap=(0.5, 0, 0), offset=(1, 0, 0)),
        #     batch_size=8,
        #     preprocessing=lambda x: x.to(self.device),
        #     postprocessing=lambda x: postprocessing(x).to(torch.device("cpu")),
        #     value_dtype=torch.complex64,
        # )

        # nufft_op, nufft_adj_op = generate_nufft_op(nufft_im_size)
        # self.nufft_cse = torch.vmap(nufft_op)  # image: b ch d h w; ktraj: b 2 len
        # self.nufft_adj_cse = torch.vmap(nufft_adj_op)  # kdata: b ch d len
        # self.nufft = torch.vmap(torch.vmap(nufft_op))
        # # image: b ph ch d h w; ktraj: b ph 2 len
        # self.nufft_adj = torch.vmap(torch.vmap(nufft_adj_op))
        # # kdata: b ph ch d len

    def training_step(self, batch, batch_idx):
        recon_opt = self.optimizers()
        recon_opt.zero_grad()
        batch = batch[0]
        # kspace data is in the shape of [ch, ph, sp]
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
        kspace_data_compensated_odd, kspace_data_compensated_even = (
            batch["kspace_data_compensated_odd"],
            batch["kspace_data_compensated_even"],
        )
        kspace_data_cse_odd, kspace_data_cse_even = (
            batch["kspace_data_cse_odd"],
            batch["kspace_data_cse_even"],
        )
        # # kspace weighted loss
        sp, len = 15, 640
        weight = torch.arange(1, len // 2 + 1, device=kspace_data_odd.device)
        weight_reverse_sample_density = torch.cat(
            [weight.flip(0), weight], dim=0
        ).expand(sp, len)
        weight_reverse_sample_density = rearrange(
            weight_reverse_sample_density, "sp len -> (sp len)"
        )

        # image_init_odd_ch = self.nufft_adj_forward(
        #     kspace_data_cse_odd.unsqueeze(0), kspace_traj_cse_odd.unsqueeze(0)
        # )
        image_init_odd_ch = nufft_adj_2d(
            kspace_data_cse_odd, kspace_traj_cse_odd, self.nufft_im_size
        )
        # shape is [1, ch, d, h, w]

        csm_odd = self.cse_forward(image_init_odd_ch).expand(5, -1, -1, -1, -1)
        # csm_smooth_loss = self.smooth_loss_coef * self.smooth_loss_fn(csm_fixed)
        # self.log_dict({"recon/csm_smooth_loss": csm_smooth_loss})

        # image_init_odd = self.nufft_adj_forward(
        #     kspace_data_compensated_odd, kspace_traj_odd
        # )
        image_init_odd = nufft_adj_2d(
            kspace_data_compensated_odd, kspace_traj_odd, self.nufft_im_size
        )
        image_init_odd = torch.sum(image_init_odd * csm_odd.conj(), dim=1)
        # shape is [ph, d, h, w]
        image_recon_odd = self.recon_module(image_init_odd.unsqueeze(0)).squeeze(0)
        loss_o2e = self.calculate_recon_loss(
            image_recon=image_recon_odd.unsqueeze(1).expand_as(csm_odd),
            csm=csm_odd,
            kspace_traj=kspace_traj_even,
            kspace_data=kspace_data_even,
            weight=weight_reverse_sample_density,
        )
        self.manual_backward(loss_o2e, retain_graph=True)
        # self.manual_backward(loss_f2m+csm_smooth_loss, retain_graph=True)
        self.log_dict({"recon/recon_loss": loss_o2e})

        # image_init_even_ch = self.nufft_adj_forward(
        #     kspace_data_cse_even.unsqueeze(0), kspace_traj_cse_even.unsqueeze(0)
        # )
        image_init_even_ch = nufft_adj_2d(
            kspace_data_cse_even, kspace_traj_cse_even, self.nufft_im_size
        )
        # if self.global_step % 4 == 0:
        #     for ch in [0, 3, 5]:
        #         to_png(self.trainer.default_root_dir+f'/image_init_moved_ch{ch}.png',
        #                image_init_moved_ch[0, ch, 0, :, :])  # , vmin=0, vmax=2)

        csm_even = self.cse_forward(image_init_even_ch).expand(5, -1, -1, -1, -1)
        # csm_moved = self.cse_forward(image_init_moved_ch)
        # csm_smooth_loss = self.smooth_loss_coef * self.smooth_loss_fn(csm_moved)

        image_init_even = nufft_adj_2d(
            kspace_data_compensated_even, kspace_traj_even, self.nufft_im_size
        )

        image_init_even = torch.sum(image_init_even * csm_even.conj(), dim=1)
        # shape is [ph, h, w]
        image_recon_even = self.recon_module(image_init_even.unsqueeze(0)).squeeze(
            0
        )  # shape is [ph, h, w]
        loss_e2o = self.calculate_recon_loss(
            image_recon=image_recon_even.unsqueeze(1).expand_as(csm_even),
            csm=csm_even,
            kspace_traj=kspace_traj_odd,
            kspace_data=kspace_data_odd,
            weight=weight_reverse_sample_density,
        )
        self.manual_backward(loss_e2o, retain_graph=True)
        # self.manual_backward(loss_m2f + csm_smooth_loss, retain_graph=True)

        if self.global_step % 4 == 0:
            for ch in [0, 3, 5]:
                to_png(
                    self.trainer.default_root_dir + f"/image_init_moved_ch{ch}.png",
                    image_init_even_ch[0, ch, 0, :, :],
                )  # , vmin=0, vmax=2)
                to_png(
                    self.trainer.default_root_dir + f"/csm_moved_ch{ch}.png",
                    csm_even[0, ch, 0, :, :],
                )  # , vmin=0, vmax=2)
            for i in range(image_init_even.shape[0]):
                to_png(
                    self.trainer.default_root_dir + f"/image_init_ph{i}.png",
                    image_init_even[i, 0, :, :],
                )  # , vmin=0, vmax=2)
                to_png(
                    self.trainer.default_root_dir + f"/image_recon_ph{i}.png",
                    image_recon_even[i, 0, :, :],
                )  # , vmin=0, vmax=2)
        recon_opt.step()

    def cse_forward(self, image_init_ch):
        ph, ch = image_init_ch.shape[0:2]
        image_init_ch_lr = image_init_ch.clone()
        for i in range(3):
            image_init_ch_lr = self.downsample(image_init_ch_lr)
        if ch < self.ch_pad:
            image_init_ch_lr = f.pad(
                image_init_ch_lr, (0, 0, 0, 0, 0, 0, 0, self.ch_pad - ch)
            )
        csm_lr = self.cse_module(image_init_ch_lr)
        csm_hr = csm_lr[:, :ch]
        for i in range(3):
            csm_hr = self.upsample(csm_hr)
        csm_hr_norm = csm_hr / torch.sqrt(
            torch.sum(torch.abs(csm_hr) ** 2, dim=1, keepdim=True)
        )
        return csm_hr_norm

    """
    def nufft_forward(self, image_init: PhChImage, kspace_traj: PhKspaceTraj):
        ph, ch, d, h, w = image_init.shape
        kspace_data = self.nufft_op(
            rearrange(image_init, "ph ch d h w -> ph (ch d) h w"),
            kspace_traj,
            norm="ortho",
        )
        # self.nufft(image_init, kspace_traj, norm="ortho")
        return rearrange(kspace_data, "ph (ch d) len -> ph ch d len", ch=ch)
    """

    # def nufft_adj_forward(self, kspace_data, kspace_traj):
    #     ph, ch, d, length = kspace_data.shape
    #     image = self.nufft_adj(
    #         rearrange(kspace_data, "ph ch d len -> ph (ch d) len"),
    #         kspace_traj,
    #         norm="ortho",
    #     )
    #     return rearrange(image, "ph (ch d) h w -> ph ch d h w", ch=ch)

    def calculate_recon_loss(
        self, image_recon, csm, kspace_traj, kspace_data, weight=None
    ):
        # kspace_data_estimated = self.nufft_forward(image_recon * csm, kspace_traj)
        kspace_data_estimated = nufft_2d(
            image_recon * csm, kspace_traj, self.nufft_im_size
        )

        loss_not_reduced = self.recon_loss_fn(
            torch.view_as_real(weight * kspace_data_estimated),
            torch.view_as_real(kspace_data * weight),
        )
        loss = torch.mean(loss_not_reduced)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        b = batch[0]

        def plot_and_validation(
            kspace_data, kspace_traj, kspace_data_cse, kspace_traj_cse
        ):
            image_recon, image_init, csm = self.forward_contrast(
                kspace_data,
                kspace_traj,
                kspace_data_cse.unsqueeze(0),
                kspace_traj_cse,
                storage_device=torch.device("cpu"),
            )
            zarr.save(
                self.trainer.default_root_dir
                + f"/epoch_{self.trainer.current_epoch}"
                + "/image_init.zarr",
                image_init[:, 35:45].abs().numpy(force=True),
            )
            zarr.save(
                self.trainer.default_root_dir
                + f"/epoch_{self.trainer.current_epoch}"
                + "/image_recon.zarr",
                image_recon[:, 35:45].abs().numpy(force=True),
            )
            zarr.save(
                self.trainer.default_root_dir
                + f"/epoch_{self.trainer.current_epoch}"
                + "/csm.zarr",
                csm[:, :, 35:45].abs().numpy(force=True),
            )
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
                    csm[0, ch, 40, :, :],
                )  # , vmin=0, vmax=2)
            for i in range(image_init.shape[0]):
                to_png(
                    self.trainer.default_root_dir
                    + f"/epoch_{self.trainer.current_epoch}"
                    + f"/image_init_ph{i}.png",
                    image_init[i, 40, :, :],
                )  # , vmin=0, vmax=2)
                to_png(
                    self.trainer.default_root_dir
                    + f"/epoch_{self.trainer.current_epoch}"
                    + f"/image_recon_ph{i}.png",
                    image_recon[i, 40, :, :],
                )  # , vmin=0, vmax=2)
            return image_recon

        return for_vmap(plot_and_validation, (0, 0, 0, 0), None, None)(
            b["kspace_data_compensated"],
            b["kspace_traj"],
            b["kspace_data_cse"],
            b["kspace_traj_cse"],
        )

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:
        xarray_ds = batch[0]
        print(xarray_ds.attrs["id"])

        def _predict(kspace_data, kspace_traj, kspace_data_cse, kspace_traj_cse):
            image_recon, image_init, csm = self.forward_contrast(
                kspace_data,
                kspace_traj,
                kspace_data_cse.unsqueeze(0),
                kspace_traj_cse,
                storage_device=torch.device("cpu"),
            )
            # xarray_ds["P2PCSE_odd"][]
            return image_recon

        def _save(data, key):
            xarray_ds[key] = xr.Variable(
                ["t", "ph", "z", "h", "w"], data.numpy(force=True)
            ).chunk(
                {
                    "t": 1,
                    "ph": -1,
                    "z": 1,
                    "h": -1,
                    "w": -1,
                }
            )
            xarray_ds.to_zarr(xarray_ds.encoding["source"], mode="a")

        image_recon = for_vmap(
            _predict,
            (0, 0, 0, 0),
            0,
            None,
        )(
            torch.from_numpy(xarray_ds["kspace_data_compensated_odd"].values),
            torch.from_numpy(xarray_ds["kspace_traj_odd"].values),
            torch.from_numpy(xarray_ds["kspace_data_cse_odd"].values),
            torch.from_numpy(xarray_ds["kspace_traj_cse_odd"].values),
        )
        print(image_recon.shape)
        # p_odd = multi_processing_save_data(image_recon, lambda x: _save(x, "P2PCSE_odd"))
        _save(image_recon, "P2PCSE_odd")
        image_recon = for_vmap(
            _predict,
            (0, 0, 0, 0),
            0,
            None,
        )(
            torch.from_numpy(xarray_ds["kspace_data_compensated_even"].values),
            torch.from_numpy(xarray_ds["kspace_traj_even"].values),
            torch.from_numpy(xarray_ds["kspace_data_cse_even"].values),
            torch.from_numpy(xarray_ds["kspace_traj_cse_even"].values),
        )
        print(image_recon.shape)
        _save(image_recon, "P2PCSE_even")
        # p_even = multi_processing_save_data(image_recon, lambda x: _save(x, "P2PCSE_even"))

    def on_predict_epoch_end(self) -> None:
        return super().on_predict_epoch_end()

    def test_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:
        xarray_ds = batch[0]
        print(xarray_ds.attrs["id"])

        def _predict(kspace_data, kspace_traj, kspace_data_cse, kspace_traj_cse):
            image_recon, image_init, csm = self.forward_contrast(
                kspace_data,
                kspace_traj,
                kspace_data_cse.unsqueeze(0),
                kspace_traj_cse,
                storage_device=torch.device("cpu"),
            )
            # xarray_ds["P2PCSE_odd"][]
            return image_recon

        def _save(data):
            ds = xr.Dataset(
                {
                    "P2PCSE": xr.Variable(
                        data=data.numpy(force=True), dims=["t", "ph", "z", "h", "w"]
                    )
                }
            ).chunk(
                {
                    "t": 1,
                    "ph": -1,
                    "z": 1,
                    "h": -1,
                    "w": -1,
                }
            )
            val_folder = Path(xarray_ds.encoding["source"]).parent
            ds.to_zarr(val_folder / (xarray_ds.attrs["id"] + "_P2PCSE.zarr"), mode="a")

        image_recon = for_vmap(
            _predict,
            (0, 0, 0, 0),
            0,
            None,
        )(
            torch.from_numpy(xarray_ds["kspace_data_compensated"].values),
            torch.from_numpy(xarray_ds["kspace_traj"].values),
            torch.from_numpy(xarray_ds["kspace_data_cse"].values),
            torch.from_numpy(xarray_ds["kspace_traj_cse"].values),
        )
        print(image_recon.shape)
        # p_odd = multi_processing_save_data(image_recon, lambda x: _save(x, "P2PCSE_odd"))
        _save(image_recon)

    @overload
    def forward(
        self,
        kspace_data: Shaped[KspaceData, "ph ch z"],
        kspace_traj: Shaped[KspaceTraj, "ph"],
        kspace_data_cse: Shaped[KspaceData, "ch z"],
        kspace_traj_cse: KspaceTraj,
        storage_device=torch.device("cpu"),
    ):
        """
        kspace_data: [ph, ch, z, len]
        kspace_traj: [ph, 2, len]
        forward for one full contrast image
        """
        image_recon = infer(
            {
                "kspace_data": kspace_data,
                "kspace_traj": kspace_traj,
                "kspace_data_cse": kspace_data_cse,
                "kspace_traj_cse": kspace_traj_cse,
            },
            self.forward,
            {
                "kspace_data": [2],
                "kspace_traj": None,
                "kspace_data_cse": [1],
                "kspace_traj_cse": None,
            },
            {
                "kspace_data": [self.patch_size[0]],
                "kspace_traj": None,
                "kspace_data_cse": [self.patch_size[0]],
                "kspace_traj_cse": None,
            },
            {
                "kspace_data": [0.5],
                "kspace_traj": None,
                "kspace_data_cse": [0.5],
                "kspace_traj_cse": None,
            },
            split_func=split_tensor,
            filter_func=cutoff_filter,
            storage_device=storage_device,
            device=self.device,
        )
        return image_recon

    @overload
    def forward(
        self,
        kspace_data: Shaped[KspaceData, "ph ch z"],
        kspace_traj: Shaped[KspaceTraj, "ph"],
        kspace_data_cse: Shaped[KspaceData, "ch z"],
        kspace_traj_cse: KspaceTraj,
    ):
        """
        kspace_data: [ph, ch, z, len]
        kspace_traj: [ph, 2, len]
        forward for one contrast patch
        """
        csm = nufft_adj_2d(kspace_data_cse, kspace_traj_cse, self.nufft_im_size)
        csm = self.cse_forward(csm)
        image_init_ch = nufft_adj_2d(kspace_data, kspace_traj, self.nufft_im_size)
        image_init = torch.sum(image_init_ch * csm.conj(), dim=1)
        image_recon = self.recon_module(image_init.unsqueeze(0)).squeeze(0)
        return image_recon, image_init, csm

    @dispatch
    def forward(
        self,
        kspace_data,
        kspace_traj,
        kspace_data_cse,
        kspace_traj_cse,
        storage_device=torch.device("cpu"),
    ):
        pass

    def configure_optimizers(self):
        recon_optimizer = self.recon_optimizer(
            [
                {"params": self.recon_module.parameters()},
                {"params": self.cse_module.parameters()},
            ],
            lr=self.recon_lr,
        )
        return recon_optimizer


# def nufft_adj_gpu(
#     kspace_data,
#     kspace_traj,
#     nufft_adj,
#     csm=None,
#     inference_device=torch.device("cuda"),
#     storage_device=torch.device("cpu"),
# ):
#     image_init = nufft_adj(
#         kspace_data.to(inference_device), kspace_traj.to(inference_device)
#     )
#     if csm is not None:
#         result_reduced = torch.sum(image_init * csm.conj().to(inference_device), dim=1)
#         return result_reduced.to(storage_device)
#     else:
#         return image_init.to(storage_device)
# def postprocessing(x):
#     x[:, :, [0, 3], ...] = 0
#     return 2 * x  # compensate for avg merger from monai

# def gradient_loss(s, penalty="l2", reduction="mean"):
#     if s.ndim != 4:
#         raise RuntimeError(f"Expected input `s` to be an 4D tensor, but got {s.shape}")
#     dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
#     dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])
#     if penalty == "l2":
#         dy = dy * dy
#         dx = dx * dx
#     elif penalty == "l1":
#         pass
#     else:
#         raise NotImplementedError
#     if reduction == "mean":
#         d = torch.mean(dx) + torch.mean(dy)
#     elif reduction == "sum":
#         d = torch.sum(dx) + torch.sum(dy)
#     return d / 2.0
