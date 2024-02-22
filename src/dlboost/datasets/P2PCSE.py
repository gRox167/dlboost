# %%
import os
import random
from glob import glob
from pathlib import Path

import torch
import xarray as xr
import numpy as np
import zarr

# from dataclasses import dataclass
from dlboost.datasets.boilerplate import recon_one_scan

# from einops import rearrange
from lightning.pytorch import LightningDataModule
from mrboost import io_utils as iou
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import dask


# %%
class DCE_P2PCSE_KXKYZ(LightningDataModule):
    def __init__(
        self,
        data_dir: os.PathLike = "/data/anlab/RawData_MR/",
        dat_file_path_list: list = [],
        cache_dir: os.PathLike = Path("/data-local/anlab/Chunxu/.cache"),
        n_splits: int = 6,
        fold_idx: int = 0,
        patch_size=(20, 320, 320),
        patch_sample_number=10,
        train_batch_size: int = 4,
        eval_batch_size: int = 1,
        num_workers: int = 0,
    ):
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

        self.dat_file_path_list = dat_file_path_list
        self.patient_ids = [
            (self.data_dir / p).parent.name for p in self.dat_file_path_list
        ]

        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.train_idx, self.val_idx = [
            (train, test) for train, test in self.kf.split(self.dat_file_path_list)
        ][self.fold_idx]

    def prepare_data(self):
        # self.train_save_path = self.cache_dir / str(self.fold_idx) / "train"
        self.train_save_path = self.cache_dir / "train"
        train_integrate = False
        if os.path.exists(self.train_save_path):
            if len(glob(str(self.train_save_path / "*.zarr"))) == len(self.patient_ids):
                train_integrate = True
        if not train_integrate:
            self.generate_train_dataset(self.train_save_path)

        # self.val_save_path = self.cache_dir / str(self.fold_idx) / "val"
        self.val_save_path = self.cache_dir / "val"
        val_integrate = False
        if os.path.exists(self.val_save_path):
            if len(glob(str(self.val_save_path / "*.zarr"))) == len(self.patient_ids):
                val_integrate = True
        if not val_integrate:
            self.generate_val_dataset(self.val_save_path)

    def generate_val_dataset(self, val_save_path):
        for p, patient_id in zip(self.dat_file_path_list, self.patient_ids):
            dat_file_to_recon = Path(self.data_dir) / p
            if os.path.exists(val_save_path / (patient_id + ".zarr")):
                continue
            raw_data = recon_one_scan(
                dat_file_to_recon, phase_num=5, time_per_contrast=10
            )
            if os.path.exists(val_save_path / (patient_id + ".zarr")):
                continue
            ds = xr.Dataset(
                data_vars=dict(
                    kspace_data=xr.Variable(
                        ["t", "ph", "ch", "z", "sp", "lens"],
                        raw_data["kspace_data_z"].numpy(),
                    ),
                    kspace_data_compensated=xr.Variable(
                        ["t", "ph", "ch", "z", "sp", "lens"],
                        raw_data["kspace_data_z_compensated"].numpy(),
                    ),
                    kspace_data_cse=xr.Variable(
                        ["t", "ph2", "ch", "z", "sp2", "lens2"],
                        raw_data["kspace_data_z"][..., 240:400].numpy(),
                    ),
                    kspace_traj=xr.Variable(
                        ["t", "ph", "complex", "sp", "lens"],
                        raw_data["kspace_traj"].numpy(),
                    ),
                    kspace_traj_cse=xr.Variable(
                        ["t", "ph2", "complex", "sp2", "lens2"],
                        raw_data["kspace_traj"][..., 240:400].numpy(),
                    ),
                    cse=xr.Variable(["ch", "z", "h", "w"], raw_data["cse"].numpy()),
                ),
                attrs={"id": patient_id},
            )
            ds = (
                ds.stack({"k": ["sp", "lens"]})
                .stack({"k2": ["ph2", "sp2", "lens2"]})
                .chunk(
                    {
                        "t": 1,
                        "ph": -1,
                        "ch": -1,
                        "z": 1,
                        "k": -1,
                        "k2": -1,
                        "complex": -1,
                        "h": -1,
                        "w": -1,
                    }
                )
            )
            ds = ds.reset_index(list(ds.indexes))
            ds.to_zarr(val_save_path / (patient_id + ".zarr"))

    def generate_train_dataset(self, train_save_path):
        # iou.check_mk_dirs(self.cache_dir / str(self.fold_idx))
        for p, patient_id in zip(self.dat_file_path_list, self.patient_ids):
            dat_file_to_recon = Path(self.data_dir) / p
            if os.path.exists(train_save_path / (patient_id + ".zarr")):
                continue
            raw_data = recon_one_scan(
                dat_file_to_recon, phase_num=10, time_per_contrast=20
            )
            ds = xr.Dataset(
                data_vars=dict(
                    kspace_data_odd=xr.Variable(
                        ["t", "ph", "ch", "z", "sp", "lens"],
                        raw_data["kspace_data_z"][:, 0::2].numpy(),
                    ),
                    kspace_data_even=xr.Variable(
                        ["t", "ph", "ch", "z", "sp", "lens"],
                        raw_data["kspace_data_z"][:, 1::2].numpy(),
                    ),
                    kspace_data_compensated_odd=xr.Variable(
                        ["t", "ph", "ch", "z", "sp", "lens"],
                        raw_data["kspace_data_z_compensated"][:, 0::2].numpy(),
                    ),
                    kspace_data_compensated_even=xr.Variable(
                        ["t", "ph", "ch", "z", "sp", "lens"],
                        raw_data["kspace_data_z_compensated"][:, 1::2].numpy(),
                    ),
                    kspace_data_cse_odd=xr.Variable(
                        ["t", "ph2", "ch", "z", "sp2", "lens2"],
                        raw_data["kspace_data_z"][:, 0::2, ..., 240:400].numpy(),
                    ),
                    kspace_data_cse_even=xr.Variable(
                        ["t", "ph2", "ch", "z", "sp2", "lens2"],
                        raw_data["kspace_data_z"][:, 1::2, ..., 240:400].numpy(),
                    ),
                    kspace_traj_odd=xr.Variable(
                        ["t", "ph", "complex", "sp", "lens"],
                        raw_data["kspace_traj"][:, 0::2].numpy(),
                    ),
                    kspace_traj_even=xr.Variable(
                        ["t", "ph", "complex", "sp", "lens"],
                        raw_data["kspace_traj"][:, 1::2].numpy(),
                    ),
                    kspace_traj_cse_odd=xr.Variable(
                        ["t", "ph2", "complex", "sp2", "lens2"],
                        raw_data["kspace_traj"][:, 0::2, ..., 240:400].numpy(),
                    ),
                    kspace_traj_cse_even=xr.Variable(
                        ["t", "ph2", "complex", "sp2", "lens2"],
                        raw_data["kspace_traj"][:, 1::2, ..., 240:400].numpy(),
                    ),
                    cse=xr.Variable(["ch", "z", "h", "w"], raw_data["cse"].numpy()),
                ),
                attrs={"id": patient_id},
            )
            ds = (
                ds.stack({"k": ["sp", "lens"]})
                .stack({"k2": ["ph2", "sp2", "lens2"]})
                .chunk(
                    {
                        "t": 1,
                        "ph": -1,
                        "ch": -1,
                        "z": 1,
                        "k": -1,
                        "k2": -1,
                        "complex": -1,
                        "h": -1,
                        "w": -1,
                    }
                )
            )
            ds = ds.reset_index(list(ds.indexes))
            ds.to_zarr(train_save_path / (patient_id + ".zarr"))

    def setup(self, init=False, stage: str = "fit"):
        dask.config.set(scheduler="synchronous")
        train_ds_list = [
            str(self.cache_dir / "train" / f"{pid}.zarr")
            for pid in self.patient_ids
        ]
        if stage == "fit":
            sample = xr.open_zarr(train_ds_list[0])
            t = sample.sizes["t"]
            z = sample.sizes["z"]
            t_indices = [i for i in range(t)]
            # z axis have z slices, we want to randomly sample n patches from these slices, each patch have p slices.
            z_indices = [
                slice(start_idx, self.patch_size[0] + start_idx)
                for start_idx in random.sample(
                    range(z - self.patch_size[0]), self.patch_sample_number
                )
            ]
            self.train_dp = [
                xr.open_zarr(train_ds).isel(t=t_idx, z=z_idx)
                for train_ds in train_ds_list
                for t_idx in t_indices
                for z_idx in z_indices
            ]
        else:
            self.train_dp = [
                xr.open_zarr(ds) for ds in train_ds_list
            ]

        val_ds_list = [
            str(self.cache_dir / "val" / f"{pid}.zarr")
            # for pid in self.patient_ids[self.val_idx.tolist()]
            for pid in self.patient_ids
        ]
        self.val_dp = [
            xr.open_zarr(ds).isel(t=slice(0, 1)) for ds in val_ds_list[2:3]
        ]
        self.pred_dp = [
            xr.open_zarr(ds) for ds in train_ds_list
        ]
        self.test_dp = [
            xr.open_zarr(ds) for ds in val_ds_list
        ]

    def train_dataloader(self):
        return DataLoader(
            self.train_dp,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=False,
            collate_fn=lambda batch_list: [
                {k: torch.from_numpy(v.to_numpy()) for k, v in x.data_vars.items()}
                for x in batch_list
            ],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dp,
            batch_size=self.val_batch_size,
            num_workers=0,
            pin_memory=False,
            collate_fn=lambda batch_list: [
                {k: torch.from_numpy(v.to_numpy()) for k, v in x.data_vars.items()}
                for x in batch_list
            ],
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_dp,
            batch_size=self.val_batch_size,
            num_workers=0,
            pin_memory=False,
            # collate_fn=lambda batch_list: [
            #     (x, {k: torch.from_numpy(v.to_numpy()) for k, v in x.data_vars.items()})
            #     for x in batch_list
            # ],
            collate_fn= lambda x: x
            )

    def test_dataloader(self):
        return DataLoader(
            self.test_dp,
            batch_size=self.val_batch_size,
            num_workers=0,
            pin_memory=False,
            # collate_fn=lambda batch_list: [
            #     (x, {k: torch.from_numpy(v.to_numpy()) for k, v in x.data_vars.items()})
            #     for x in batch_list
            # ],
            collate_fn= lambda x: x
            )

    def transfer_batch_to_device(
        self, batch, device: torch.device, dataloader_idx: int
    ):
        if self.trainer.training:
            return super().transfer_batch_to_device(batch, device, dataloader_idx)
        else:
            return batch


# %%
if __name__ == "__main__":
    # dataset = LibriSpeech()
    # dataset.prepare_data()
    data = DCE_P2PCSE_KXKYZ()
    data.prepare_data()
    data.setup()

# %%
