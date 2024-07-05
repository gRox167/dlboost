from pathlib import Path
from typing import Any, Sequence

import dask.array as da
import lightning as L
import numpy as np
import torch
import xarray as xr
import zarr
from dask.distributed import Client
from dlboost.models import ComplexUnet, DWUNet
from dlboost.utils import to_png
from dlboost.utils.io_utils import async_save_xarray_dataset
from dlboost.utils.patch_utils import cutoff_filter, infer
from dlboost.utils.tensor_utils import for_vmap, interpolate
from einops import rearrange  # , reduce, repeat
from jaxtyping import Shaped
from mrboost.computation import nufft_2d, nufft_adj_2d
from optree import PyTree, tree_map
from plum import activate_autoreload

activate_autoreload()
from dlboost.utils.type_utils import (
    ComplexImage2D,
    ComplexImage3D,
    KspaceData,
    KspaceTraj,
)
from icecream import ic
from plum import dispatch, overload
from torch import Tensor, optim
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
            ignore=[
                "recon_module",
                "cse_module",
                "regis_module",
                "recon_loss_fn",
                "client",
            ]
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

        
        # kspace weighted loss
        # sp, len = 15, 640
        # weight = torch.arange(1, len // 2 + 1, device=kspace_data_odd.device)
        # weight_reverse_sample_density = torch.cat(
        #     [weight.flip(0), weight], dim=0
        # ).expand(sp, len)
        # weight_reverse_sample_density = rearrange(
        #     weight_reverse_sample_density, "sp len -> (sp len)"
        # )

        image_recon_odd, image_init_odd, csm_odd = self.forward(
            kspace_data_compensated_odd, kspace_traj_odd, kspace_data_cse_odd, kspace_traj_cse_odd
        )
        
        loss_o2e = self.calculate_recon_loss(
            image_recon=image_recon_odd,
            csm=csm_odd,
            kspace_traj=kspace_traj_even,
            kspace_data=kspace_data_even,
            # weight=weight_reverse_sample_density,
        )
        self.manual_backward(loss_o2e, retain_graph=True)
        # self.manual_backward(loss_f2m+csm_smooth_loss, retain_graph=True)
        self.log_dict({"recon/recon_loss": loss_o2e})

        image_recon_even, image_init_even, csm_even = self.forward(
            kspace_data_compensated_even,
            kspace_traj_even,
            kspace_data_cse_even,
            kspace_traj_cse_even,
        )

        loss_e2o = self.calculate_recon_loss(
            image_recon=image_recon_even,
            csm=csm_even,
            kspace_traj=kspace_traj_odd,
            kspace_data=kspace_data_odd,
            # weight=weight_reverse_sample_density,
        )
        self.manual_backward(loss_e2o, retain_graph=True)
        # self.manual_backward(loss_m2f + csm_smooth_loss, retain_graph=True)

        if self.global_step % 4 == 0:
            for ch in [0, 3, 5]:
                # to_png(
                #     self.trainer.default_root_dir + f"/image_init_moved_ch{ch}.png",
                #     image_init_even_ch[0, ch, 0, :, :],
                # )  # , vmin=0, vmax=2)
                to_png(
                    self.trainer.default_root_dir + f"/csm_moved_ch{ch}.png",
                    csm_even[ch, 0, :, :],
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

    def calculate_recon_loss(
        self, image_recon, csm, kspace_traj, kspace_data, weight=None
    ):
        # kspace_data_estimated = self.nufft_forward(image_recon * csm, kspace_traj)
        ch = csm.shape[0]
        kspace_data_estimated = nufft_2d(
            image_recon.unsqueeze(1).expand(-1,ch,-1,-1,-1) * csm, kspace_traj, self.nufft_im_size
        )
        kspace_data_estimated_detatched = kspace_data_estimated.detach().abs()
        norm_factor = kspace_data_estimated_detatched.max()
        weight = 1 / ( kspace_data_estimated_detatched/norm_factor + 1e-5)
        # weight /= weight.max()
        loss_not_reduced = self.recon_loss_fn(
            torch.view_as_real(kspace_data_estimated * weight),
            torch.view_as_real(kspace_data * weight),
        )
        loss = torch.mean(loss_not_reduced)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        b = batch[0]
        ch = b["kspace_data_compensated"].sizes["ch"]
        input_tree = (
            b["kspace_data_compensated"].isel(t=0),
            b["kspace_traj"].isel(t=0),
            b["kspace_data_cse"].isel(t=0),
            b["kspace_traj_cse"].isel(t=0),
        )
        # currently beartype doesn't support deep dict typing
        with torch.no_grad():
            output_tree = infer(
                input_tree,
                (
                    {"ph": 5, "z": 10, "h": 320, "w": 320},
                    {"ph": 5, "z": 10, "h": 320, "w": 320},
                    {"ch": ch, "z": 10, "h": 320, "w": 320},
                ),
                self.forward,
                {"z": 4},
                {"z": 0.25},
                cutoff_filter,
                self.device,
                "xarray",
                output_dtype=np.complex64,
            )

        image_recon, image_init, cse = output_tree
        ds = xr.Dataset(
            {
                "P2PCSE": image_recon,
                "image_init": image_init,
                "csm": cse,
            }
        ).chunk({"ch": -1, "ph": -1, "z": 1, "h": -1, "w": -1})
        save_path = (
            self.trainer.default_root_dir
            + f"/epoch_{self.trainer.current_epoch}/val.zarr"
        )
        print(save_path)
        ds.to_zarr(save_path, mode="w")

        for i in range(image_init.shape[0]):
            to_png(
                self.trainer.default_root_dir
                + f"/epoch_{self.trainer.current_epoch}"
                + f"/image_init_ph{i}.png",
                image_init[i, 5, :, :],
            )  # , vmin=0, vmax=2)
            to_png(
                self.trainer.default_root_dir
                + f"/epoch_{self.trainer.current_epoch}"
                + f"/image_recon_ph{i}.png",
                image_recon[i, 5, :, :],
            )  # , vmin=0, vmax=2)
        # future = async_save_xarray_dataset(ds, save_path, self.client)
        # self.validation_step_outputs.append(future)
        # return future

        # def plot_and_validation(
        #     kspace_data, kspace_traj, kspace_data_cse, kspace_traj_cse
        # ):
        #     image_recon, image_init, csm = self.forward_contrast(
        #         kspace_data,
        #         kspace_traj,
        #         kspace_data_cse.unsqueeze(0),
        #         kspace_traj_cse,
        #         storage_device=torch.device("cpu"),
        #     )
        #     zarr.save(
        #         self.trainer.default_root_dir
        #         + f"/epoch_{self.trainer.current_epoch}"
        #         + "/image_init.zarr",
        #         image_init[:, 35:45].abs().numpy(force=True),
        #     )
        #     zarr.save(
        #         self.trainer.default_root_dir
        #         + f"/epoch_{self.trainer.current_epoch}"
        #         + "/image_recon.zarr",
        #         image_recon[:, 35:45].abs().numpy(force=True),
        #     )
        #     zarr.save(
        #         self.trainer.default_root_dir
        #         + f"/epoch_{self.trainer.current_epoch}"
        #         + "/csm.zarr",
        #         csm[:, :, 35:45].abs().numpy(force=True),
        #     )
        #     print(
        #         "Save image_init, image_recon, csm to "
        #         + self.trainer.default_root_dir
        #         + f"/epoch_{self.trainer.current_epoch}"
        #     )
        #     for ch in [0, 3, 5]:
        #         to_png(
        #             self.trainer.default_root_dir
        #             + f"/epoch_{self.trainer.current_epoch}"
        #             + f"/csm_moved_ch{ch}.png",
        #             csm[0, ch, 40, :, :],
        #         )  # , vmin=0, vmax=2)
        #     for i in range(image_init.shape[0]):
        #         to_png(
        #             self.trainer.default_root_dir
        #             + f"/epoch_{self.trainer.current_epoch}"
        #             + f"/image_init_ph{i}.png",
        #             image_init[i, 40, :, :],
        #         )  # , vmin=0, vmax=2)
        #         to_png(
        #             self.trainer.default_root_dir
        #             + f"/epoch_{self.trainer.current_epoch}"
        #             + f"/image_recon_ph{i}.png",
        #             image_recon[i, 40, :, :],
        #         )  # , vmin=0, vmax=2)
        #     return image_recon

        # return for_vmap(plot_and_validation, (0, 0, 0, 0), None, None)(
        #     b["kspace_data_compensated"],
        #     b["kspace_traj"],
        #     b["kspace_data_cse"],
        #     b["kspace_traj_cse"],
        # )

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def on_validation_epoch_end(self):
        for future in self.validation_step_outputs:
            future.result()
        self.validation_step_outputs.clear()  # free memory

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:
        xarray_ds = batch[0]
        ch = xarray_ds["kspace_data_compensated_odd"].sizes["ch"]
        t = xarray_ds["kspace_data_compensated_odd"].sizes["t"]

        def predict(ds, phase="odd"):
            input_tree = (
                ds[f"kspace_data_compensated_{phase}"],
                ds[f"kspace_traj_{phase}"],
                ds[f"kspace_data_cse_{phase}"],
                ds[f"kspace_traj_cse_{phase}"],
            )
            ic(input_tree[0].sizes)
            with torch.no_grad():
                output_tree = infer(
                    input_tree,
                    (
                        {"ph": 5, "z": 80, "h": 320, "w": 320},
                        {"ph": 5, "z": 80, "h": 320, "w": 320},
                        {"ch": ch, "z": 80, "h": 320, "w": 320},
                    ),
                    self.forward,
                    {"z": 4},
                    {"z": 0.25},
                    cutoff_filter,
                    self.device,
                    "xarray",
                    output_dtype=np.complex64,
                )
            image_recon, image_init, cse = output_tree
            image_recon_contrast = image_recon.expand_dims("t", 0)
            # image_recon_contrast = xr.DataArray(
            #     np.random.rand(1, 5, 80, 320, 320),
            #     dims=["t", "ph", "z", "h", "w"],
            # )
            return image_recon_contrast

        result_ds = xr.Dataset(
            {
                "odd": xr.DataArray(
                    da.zeros((17, 5, 80, 320, 320), chunks=(1, 5, 1, 320, 320)),
                    dims=["t", "ph", "z", "h", "w"],
                ),
                "even": xr.DataArray(
                    da.zeros((17, 5, 80, 320, 320), chunks=(1, 5, 1, 320, 320)),
                    dims=["t", "ph", "z", "h", "w"],
                ),
            }
        )
        result_path = (
            str(
                Path(xarray_ds.encoding["source"]).parent
                / "P2PCSE"
                / xarray_ds.attrs["id"]
            )
            + ".zarr"
        )
        # ic(result_path)
        result_ds.to_zarr(result_path, compute=False)
        for i in range(t):
            image_recon_odd = predict(xarray_ds.isel({"t": i}), "odd")
            image_recon_even = predict(xarray_ds.isel({"t": i}), "even")
            output_ds = xr.Dataset(
                {
                    "odd": image_recon_odd,
                    "even": image_recon_even,
                }
            )
            future = async_save_xarray_dataset(
                output_ds,
                result_path,
                self.client,
                mode="a",
                region={"t": slice(i, i + 1)},
            )
            self.predict_step_outputs.append(future)

    def on_predict_epoch_start(self) -> None:
        self.client = Client()
        self.predict_step_outputs = []

    def on_predict_epoch_end(self) -> None:
        self.client.gather(self.predict_step_outputs)
        self.predict_step_outputs.clear()  # free memory
        self.client.close()
        del self.client

    def test_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:
        xarray_ds = batch[0]
        ch = xarray_ds["kspace_data_compensated"].sizes["ch"]
        t = xarray_ds["kspace_data_compensated"].sizes["t"]

        def predict(ds):
            input_tree = (
                ds[f"kspace_data_compensated"],
                ds[f"kspace_traj"],
                ds[f"kspace_data_cse"],
                ds[f"kspace_traj_cse"],
            )
            with torch.no_grad():
                output_tree = infer(
                    input_tree,
                    (
                        {"ph": 5, "z": 80, "h": 320, "w": 320},
                        {"ph": 5, "z": 80, "h": 320, "w": 320},
                        {"ch": ch, "z": 80, "h": 320, "w": 320},
                    ),
                    self.forward,
                    {"z": 4},
                    {"z": 0.25},
                    cutoff_filter,
                    self.device,
                    "xarray",
                    output_dtype=np.complex64,
                )
            image_recon, image_init, cse = output_tree
            image_recon_contrast = image_recon.expand_dims("t", 0)
            return image_recon_contrast

        result_ds = xr.Dataset(
            {
                "image_recon": xr.DataArray(
                    da.zeros((1, 5, 80, 320, 320), chunks=(1, 5, 1, 320, 320)),
                    dims=["t", "ph", "z", "h", "w"],
                ),
            }
        )
        result_path = (
            str(
                Path(xarray_ds.encoding["source"]).parent
                / "P2PCSE"
                / xarray_ds.attrs["id"]
            )
            + ".zarr"
        )
        # ic(result_path)
        # result_ds.to_zarr(result_path, compute=False)
        result_ds.to_zarr(result_path)
        # for i in range(t):
        #     image_recon = predict(xarray_ds.isel({"t": i}))
        #     output_ds = xr.Dataset({"image_recon": image_recon})
        #     future = async_save_xarray_dataset(
        #         output_ds,
        #         result_path,
        #         self.client,
        #         mode="a",
        #         region={"t": slice(i, i + 1)},
        #     )
        #     self.test_step_outputs.append(future)

    def on_test_epoch_start(self) -> None:
        self.client = Client()
        self.test_step_outputs = []

    def on_test_epoch_end(self) -> None:
        self.client.gather(self.test_step_outputs)
        self.test_step_outputs.clear()  # free memory
        self.client.close()
        del self.client

    @overload
    def forward(self, input_tree: tuple) -> Sequence[Tensor]:
        # for i in input_tree:
        #     print(i.shape)
        output = self.forward(*(i for i in input_tree))
        return tuple(i for i in output)

    @overload
    def forward(
        self,
        kspace_data: Shaped[KspaceData, "ph ch z"],
        kspace_traj: Shaped[KspaceTraj, "ph"],
        kspace_data_cse: Shaped[KspaceData, "ch z"],
        kspace_traj_cse: KspaceTraj,
    ) -> Sequence[Tensor]:
        """
        kspace_data: [ph, ch, z, len]
        kspace_traj: [ph, 2, len]
        forward for one contrast patch
        """
        csm = nufft_adj_2d(kspace_data_cse, kspace_traj_cse, self.nufft_im_size)
        csm = self.cse_forward(csm.expand(5, -1, -1, -1, -1))
        image_init_ch = nufft_adj_2d(kspace_data, kspace_traj, self.nufft_im_size)
        image_init = torch.sum(image_init_ch * csm.conj(), dim=1)
        image_recon = self.recon_module(image_init.unsqueeze(0)).squeeze(0)
        return image_recon, image_init, csm[0]

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

    def on_load_checkpoint(self, checkpoint):
        checkpoint["optimizer_states"] = []
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
