# %%
import os
import random
from glob import glob
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import torch
import xarray as xr
from dask.distributed import Client
from icecream import ic

# from einops import rearrange
from lightning.pytorch import LightningDataModule
from sklearn.model_selection import KFold
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from dlboost.datasets.boilerplate import collate_fn

# from dataclasses import dataclass
from dlboost.NODEO.Registration import registration
from dlboost.utils.io_utils import async_save_xarray_dataset


class MOTIF_CORD_TRAIN(Dataset):
    def __init__(
        self,
        cache_dir,
        train_patient_ids,
        patch_size=(20, 320, 320),
        patch_sample_number=5,
    ):
        self.cache_dir = cache_dir
        self.train_patient_ids = train_patient_ids
        self.patch_size = patch_size
        self.patch_sample_number = patch_sample_number
        self.train_kd = [
            str(self.cache_dir / "train" / f"{pid}.zarr")
            for pid in self.train_patient_ids
        ]
        self.train_p2p = [
            xr.open_zarr(str(self.cache_dir / "train" / "P2PCSE" / f"{pid}.zarr"))
            for pid in self.train_patient_ids
        ]
        self.train_mvf = [
            xr.open_zarr(str(self.cache_dir / "train" / "MVF" / f"{pid}.zarr"))
            for pid in self.train_patient_ids
        ]
        sample = self.train_p2p[0]
        t_size = sample.sizes["t"]
        t_indices = [i for i in range(t_size)] * self.patch_sample_number
        # z axis have z slices, we want to randomly sample n patches from these slices, each patch have p slices.

        z_indices = [
            slice(start_idx, self.patch_size[0] + start_idx)
            for start_idx in random.choices(
                range(sample.sizes["z"] - self.patch_size[0]), k=len(t_indices)
            )
        ]
        self.train_idx = [
            (kd, p2p, mvf, t_idx, z_slice)
            for kd, p2p, mvf in zip(self.train_kd, self.train_p2p, self.train_mvf)
            for t_idx, z_slice in zip(t_indices, z_indices)
        ]

    def __len__(self):
        return len(self.train_idx)

    def __getitem__(self, index):
        def get_data_dict(train_kd, train_p2p, train_mvf, t_idx, z_slice):
            return {
                "kspace_data_odd": xr.open_zarr(train_kd)["kspace_data_odd"].isel(
                    t=t_idx, z=z_slice
                ),
                "kspace_data_even": xr.open_zarr(train_kd)["kspace_data_even"].isel(
                    t=t_idx, z=z_slice
                ),
                "kspace_data_cse_odd": xr.open_zarr(train_kd)[
                    "kspace_data_cse_odd"
                ].isel(t=t_idx, z=z_slice),
                "kspace_data_cse_even": xr.open_zarr(train_kd)[
                    "kspace_data_cse_even"
                ].isel(t=t_idx, z=z_slice),
                "kspace_traj_odd": xr.open_zarr(train_kd)["kspace_traj_odd"].isel(
                    t=t_idx
                ),
                "kspace_traj_even": xr.open_zarr(train_kd)["kspace_traj_even"].isel(
                    t=t_idx
                ),
                "kspace_traj_cse_odd": xr.open_zarr(train_kd)[
                    "kspace_traj_cse_odd"
                ].isel(t=t_idx),
                "kspace_traj_cse_even": xr.open_zarr(train_kd)[
                    "kspace_traj_cse_even"
                ].isel(t=t_idx),
                "P2PCSE_odd": train_p2p["P2PCSE_odd"].isel(t=t_idx, z=z_slice),
                "P2PCSE_even": train_p2p["P2PCSE_even"].isel(t=t_idx, z=z_slice),
                "MVF_odd": train_mvf["MVF_odd"]
                .isel(t=t_idx, z=z_slice)
                .astype("float32"),
                "MVF_even": train_mvf["MVF_even"]
                .isel(t=t_idx, z=z_slice)
                .astype("float32"),
            }

        return get_data_dict(*self.train_idx[index])


class MOTIF_CORD_VAL(Dataset):
    def __init__(
        self, cache_dir, patient_ids, t_slice=slice(None), z_slice=slice(None)
    ):
        self.cache_dir = cache_dir
        self.patient_ids = patient_ids
        self.val_kd = [
            xr.open_zarr(str(self.cache_dir / "train" / f"{pid}.zarr"))
            for pid in self.patient_ids
        ]
        self.val_p2p = [
            xr.open_zarr(str(self.cache_dir / "train" / "P2PCSE" / f"{pid}.zarr"))
            for pid in self.patient_ids
        ]
        self.val_mvf = [
            xr.open_zarr(str(self.cache_dir / "train" / "MVF" / f"{pid}.zarr"))
            for pid in self.patient_ids
        ]
        self.t_slice = t_slice
        self.z_slice = z_slice

    def __len__(self):
        return len(self.val_kd)

    def __getitem__(self, index):
        def get_data_dict(kd, p2p, mvf):
            return {
                "kspace_data_odd": kd["kspace_data_odd"].isel(
                    t=self.t_slice, z=self.z_slice
                ),
                "kspace_data_cse_odd": kd["kspace_data_cse_odd"].isel(
                    t=self.t_slice, z=self.z_slice
                ),
                "kspace_traj_odd": kd["kspace_traj_odd"].isel(t=self.t_slice),
                "kspace_traj_cse_odd": kd["kspace_traj_cse_odd"].isel(t=self.t_slice),
                "P2PCSE_odd": p2p["P2PCSE_odd"].isel(t=self.t_slice, z=self.z_slice),
                "MVF_odd": mvf["MVF_odd"]
                .isel(t=self.t_slice, z=self.z_slice)
                .astype("float32"),
                "path": Path(kd.encoding["source"]),
                "id": kd.attrs["id"],
            }

        return get_data_dict(
            self.val_kd[index], self.val_p2p[index], self.val_mvf[index]
        )


def registration_for_one_contrast(images):
    device = torch.device("cuda:0")
    moving = torch.from_numpy(images[:, 0:1].to_numpy()).abs().float().to(device)
    scale_factor = (1.0, 0.5, 0.5)
    moving = F.interpolate(
        moving, scale_factor=scale_factor, mode="trilinear", align_corners=True
    )
    df_list = []
    for ph in range(1, 5):
        fixed = (
            torch.from_numpy(images[:, ph : ph + 1].to_numpy()).abs().float().to(device)
        )
        fixed = F.interpolate(
            fixed, scale_factor=scale_factor, mode="trilinear", align_corners=True
        )
        df, df_with_grid, warped_moving = registration(
            device,
            moving,
            fixed,
        )
        df_list.append(df)
    df_one_contrast = torch.cat(df_list)
    return df_one_contrast


# %%
class DCE_MOTIF_KXKYZ(LightningDataModule):
    def __init__(
        self,
        data_dir: os.PathLike = "/data/anlab/RawData_MR/",
        dat_file_path_list: list = [],
        cache_dir: os.PathLike = Path("/data-local/anlab/Chunxu/.cache"),
        rand_idx: list = [],
        n_splits: int = 6,
        fold_idx: int = 0,
        patch_size=(20, 320, 320),
        patch_sample_number=5,
        train_batch_size: int = 4,
        eval_batch_size: int = 1,
        num_workers: int = 0,
    ):
        super().__init__()
        super().__init__()
        # self.train_scope = slice(*train_scope)
        # self.val_scope = slice(*val_scope)
        # self.load_scope = slice(*load_scope)
        self.patch_size = patch_size
        self.patch_sample_number = patch_sample_number

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.val_batch_size = eval_batch_size
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) / ".DCE_MOTIF_CORD"
        self.num_workers = num_workers
        self.fold_idx = fold_idx
        self.ref_phase = 0
        self.dat_file_path_list = dat_file_path_list

        self.patient_ids = [
            (self.data_dir / p).parent.name + "_DB"
            if "_Dyn_DB.dat" in p
            else (self.data_dir / p).parent.name
            for p in self.dat_file_path_list
        ]

        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        print("rand_idx: ", rand_idx)
        splits = np.array_split(rand_idx, n_splits)
        print("splits: ", splits)
        # self.val_idx = splits.pop(fold_idx)
        # self.train_idx = np.concatenate(splits)
        # self.train_patient_ids = [self.patient_ids[i] for i in self.train_idx]
        # self.val_patient_ids = [self.patient_ids[i] for i in self.val_idx]
        # print("validation patient ids: ", [self.patient_ids[i] for i in self.val_idx])
        # print("training patient ids: ", [self.patient_ids[i] for i in self.train_idx])
        # for testing
        self.train_patient_ids = self.patient_ids
        self.val_patient_ids = self.patient_ids
        ic(self.patient_ids)
        self.client = None
        self.futures = []

    def prepare_data(self):
        # self.train_save_path = self.cache_dir / str(self.fold_idx) / "train"
        # return
        self.train_save_path = self.cache_dir / "train" / "MVF"
        train_integrate = False
        if os.path.exists(self.train_save_path):
            if len(glob(str(self.train_save_path / "*.zarr"))) == len(
                self.train_patient_ids
            ) + len(self.val_patient_ids):
                train_integrate = True
        if not train_integrate:
            self.generate_train_dataset(self.train_save_path)

        # self.val_save_path = self.cache_dir / "val" / "MVF"
        # val_integrate = False
        # if os.path.exists(self.val_save_path):
        #     if len(glob(str(self.val_save_path / "*.zarr"))) == len(
        #         self.train_patient_ids
        #     ) + len(self.val_patient_ids):
        #         val_integrate = True
        # if not val_integrate:
        #     self.generate_val_dataset(self.val_save_path)

        for patient_id, i, future in self.futures:
            future.result()
            ic(patient_id, "contrast:", i, "is written")

    def generate_val_dataset(self, val_save_path):
        for p, patient_id in zip(self.dat_file_path_list, self.patient_ids):
            # dat_file_to_recon = Path(self.data_dir) / p
            save_path = val_save_path / (patient_id + ".zarr")
            if os.path.exists(save_path):
                continue
            p2p = xr.open_zarr(val_save_path.parent / "P2PCSE" / (patient_id + ".zarr"))

            result_ds = xr.Dataset(
                {
                    "MVF": xr.DataArray(
                        da.zeros(
                            (34, 4, 3, 80, 160, 160), chunks=(1, 4, 3, 1, 160, 160)
                        ),
                        dims=["t", "target_ph", "dim", "z", "h", "w"],
                    ),
                }
            )

            result_ds.to_zarr(save_path, compute=False)
            for i in range(p2p.sizes["t"]):
                df = registration_for_one_contrast(
                    p2p["P2PCSE"].isel({"t": slice(i, i + 1)})
                )
                output_ds = xr.Dataset(
                    {
                        "MVF": (
                            ("t", "target_ph", "dim", "z", "h", "w"),
                            df.unsqueeze(0).numpy(force=True),
                        ),
                    }
                )
                future = async_save_xarray_dataset(
                    output_ds,
                    save_path,
                    self.client,
                    mode="a",
                    region={"t": slice(i, i + 1)},
                )
                self.futures.append((patient_id, i, future))

    def generate_train_dataset(self, train_save_path):
        for p, patient_id in zip(self.dat_file_path_list, self.patient_ids):
            save_path = train_save_path / (patient_id + ".zarr")
            if os.path.exists(save_path):
                continue
            if self.client is None:
                self.client = Client()
            p2p = xr.open_zarr(
                train_save_path.parent / "P2PCSE" / (patient_id + ".zarr")
            )

            result_ds = xr.Dataset(
                {
                    "MVF_odd": xr.DataArray(
                        da.zeros(
                            (17, 4, 3, 80, 160, 160), chunks=(1, 4, 3, 1, 160, 160)
                        ),
                        dims=["t", "target_ph", "dim", "z", "h", "w"],
                    ),
                    "MVF_even": xr.DataArray(
                        da.zeros(
                            (17, 4, 3, 80, 160, 160), chunks=(1, 4, 3, 1, 160, 160)
                        ),
                        dims=["t", "target_ph", "dim", "z", "h", "w"],
                    ),
                }
            )

            result_ds.to_zarr(save_path, compute=False)
            for i in range(p2p.sizes["t"]):
                df_odd = registration_for_one_contrast(
                    p2p["P2PCSE_odd"].isel({"t": slice(i, i + 1)})
                )
                df_even = registration_for_one_contrast(
                    p2p["P2PCSE_even"].isel({"t": slice(i, i + 1)})
                )
                output_ds = xr.Dataset(
                    {
                        "MVF_odd": (
                            ("t", "target_ph", "dim", "z", "h", "w"),
                            df_odd.unsqueeze(0).numpy(force=True),
                        ),
                        "MVF_even": (
                            ("t", "target_ph", "dim", "z", "h", "w"),
                            df_even.unsqueeze(0).numpy(force=True),
                        ),
                    }
                )
                future = async_save_xarray_dataset(
                    output_ds,
                    save_path,
                    self.client,
                    mode="a",
                    region={"t": slice(i, i + 1)},
                )
                self.futures.append((patient_id, i, future))
                ic(patient_id, "contrast:", i, "is writting")

    def setup(self, init=False, stage: str = "fit"):
        if stage == "fit":
            dask.config.set(scheduler="synchronous")
            self.train_ds = MOTIF_CORD_TRAIN(
                self.cache_dir,
                self.train_patient_ids,
                self.patch_size,
                self.patch_sample_number,
            )
        self.val_ds = MOTIF_CORD_VAL(
            self.cache_dir, self.val_patient_ids[0:1], slice(0, 1), slice(30, 50)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            num_workers=0,
            pin_memory=True,
            collate_fn=lambda x: x,
        )

    def predict_dataloader(self):  # -> DataLoader:
        return DataLoader(
            self.val_dp,
            batch_size=self.val_batch_size,
            num_workers=0,
            pin_memory=False,
            collate_fn=lambda batch_list: [
                (x, {k: torch.from_numpy(v.to_numpy()) for k, v in x.data_vars.items()})
                for x in batch_list
            ],
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dp,
            batch_size=self.val_batch_size,
            num_workers=1,
            pin_memory=False,
            collate_fn=lambda batch_list: [
                (x, {k: torch.from_numpy(v.to_numpy()) for k, v in x.data_vars.items()})
                for x in batch_list
            ],
        )

    # def transfer_batch_to_device(
    #     self, batch, device: torch.device, dataloader_idx: int
    # ):
    #     if self.trainer.training:
    #         return super().transfer_batch_to_device(batch, device, dataloader_idx)
    #     else:
    #         return batch


# %%
if __name__ == "__main__":
    # dataset = LibriSpeech()
    # dataset.prepare_data()
    data = DCE_MOTIF_KXKYZ()
    data.prepare_data()
    data.setup()

# %%
