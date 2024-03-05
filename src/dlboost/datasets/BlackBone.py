# %%
import os
import random
from glob import glob
from pathlib import Path

import dask
import numpy as np
import torch
import xarray as xr
import zarr

# from dataclasses import dataclass
from dlboost.datasets.boilerplate import recon_one_scan

# from einops import rearrange
from lightning.pytorch import LightningDataModule
from matplotlib.pylab import f
from mrboost import computation as comp
from mrboost.reconstruction import BlackBone_LowResFrames

# from mrboost import io_utils as iou
from torch.utils.data import DataLoader


# %%
class BlackBone(LightningDataModule):
    def __init__(
        self,
        data_dir: os.PathLike = "/data/anlab/RawData_MR/",
        dat_file_path_list: list[str] = [],
        motion_curve_path_list: list[str] = [],
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
        self.patch_size = patch_size
        self.patch_sample_number = patch_sample_number

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.val_batch_size = eval_batch_size
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) / ".BlackBone"
        self.num_workers = num_workers
        self.rand_idx = rand_idx
        self.n_splits = n_splits
        print(fold_idx)
        self.fold_idx = fold_idx

        self.dat_file_path_list = dat_file_path_list
        self.motion_curve_path_list = motion_curve_path_list
        self.patient_ids = [
            (self.data_dir / p).parent.name for p in self.dat_file_path_list
        ]

        print("rand_idx: ", rand_idx)
        splits = np.array_split(rand_idx, n_splits)
        print("splits: ", splits)
        self.val_idx = splits.pop(fold_idx)
        self.train_idx = np.concatenate(splits)
        self.train_patient_ids = [self.patient_ids[i] for i in self.train_idx]
        self.val_patient_ids = [self.patient_ids[i] for i in self.val_idx]
        print("validation patient ids: ", [self.patient_ids[i] for i in self.val_idx])
        print("training patient ids: ", [self.patient_ids[i] for i in self.train_idx])

    def prepare_data(self):
        # self.train_save_path = self.cache_dir / str(self.fold_idx) / "train"
        self.train_save_path = self.cache_dir / "train"
        train_integrate = False
        if os.path.exists(self.train_save_path):
            if len(glob(str(self.train_save_path / "*.zarr"))) == len(
                self.train_idx
            ) + len(self.val_idx):
                train_integrate = True
        if not train_integrate:
            self.generate_train_dataset(self.train_save_path)

        # self.val_save_path = self.cache_dir / str(self.fold_idx) / "val"
        self.val_save_path = self.cache_dir / "val"
        val_integrate = False
        if os.path.exists(self.val_save_path):
            if len(glob(str(self.val_save_path / "*.zarr"))) == len(
                self.train_idx
            ) + len(self.val_idx):
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
        for p, m, patient_id in zip(
            self.dat_file_path_list, self.motion_curve_path_list, self.patient_ids
        ):
            dat_file_location = Path(self.data_dir) / p
            motion_curve_location = Path(self.data_dir) / m
            if os.path.exists(train_save_path / (patient_id + ".zarr")):
                continue
            r = BlackBone_LowResFrames(
                dat_file_location, motion_curve_location, device=torch.device("cuda:1")
            )
            data_raw = r.get_raw_data(r.dat_file_location)
            r.args_init()
            data_dict = r.data_preprocess(data_raw)
            frame_num = len(data_dict["kspace_data_z"])
            dims = [["ch", "z", f"sp_{i}", "lens"] for i in range(frame_num)]
            +[["complex", f"sp_{i}", "lens"] for i in range(frame_num)]
            +[[f"sp_{i}", "lens"] for i in range(frame_num)]
            +[["ch", "z", "h", "w"]]
            keys = (
                [f"kspace_data_z_{i}" for i in range(frame_num)]
                + [f"kspace_traj_{i}" for i in range(frame_num)]
                + [f"kspace_density_compensation_{i}" for i in range(frame_num)]
                + ["csm"]
            )
            values = [data_dict["kspace_data_z"][i].numpy() for i in range(frame_num)]
            +[data_dict["kspace_traj"][i].numpy() for i in range(frame_num)]
            +[
                data_dict["kspace_density_compensation"][i].numpy()
                for i in range(frame_num)
            ]
            +[data_dict["cse"].numpy()]

            variable_dict = dict()
            for d, k, v in zip(dims, keys, values):
                variable_dict[k] = xr.Variable(d, v)

            ds = xr.Dataset(
                data_vars=variable_dict,
                attrs={"id": patient_id},
            )

            chunk_dict = dict()
            for d in sum(dims):
                if d == "z":
                    chunk_dict[d] = 1
                else:
                    chunk_dict[d] = -1

            ds = ds.chunk(chunk_dict)
            ds = ds.reset_index(list(ds.indexes))
            ds.to_zarr(train_save_path / (patient_id + ".zarr"))

    def setup(self, init=False, stage: str = "fit"):
        dask.config.set(scheduler="synchronous")
        train_ds_list = [
            str(self.cache_dir / "train" / f"{pid}.zarr")
            for pid in self.train_patient_ids
        ]
        if stage == "fit":
            # str(self.cache_dir / str(self.fold_idx) / "train" / "*.zarr")
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
                # xr.open_zarr(train_ds).isel(t=t_idx)
                for train_ds in train_ds_list
                for t_idx in t_indices
                for z_idx in z_indices
            ]
        else:
            self.train_dp = [xr.open_zarr(ds) for ds in train_ds_list]

        val_ds_list = [
            (
                str(self.cache_dir / "val" / f"{pid}.zarr"),
                str((self.cache_dir / "val" / f"{pid}_P2PCSE.zarr")),
            )
            for pid in self.patient_ids[2:3]
        ]
        self.val_dp = [
            (
                xr.open_zarr(ds).isel(t=slice(0, 1), z=slice(36, 44)),
                xr.open_zarr(init).isel(t=slice(0, 1), z=slice(36, 44)),
            )
            for ds, init in val_ds_list
        ]
        # self.pred_dp = [xr.open_zarr(ds) for ds in val_ds_list]

    def train_dataloader(self):
        train_keys = [
            "kspace_data_odd",
            "kspace_data_even",
            "kspace_data_cse_odd",
            "kspace_data_cse_even",
            "kspace_traj_odd",
            "kspace_traj_even",
            "kspace_traj_cse_odd",
            "kspace_traj_cse_even",
            "P2PCSE_odd",
            "P2PCSE_even",
        ]

        def collate_fn(batch_list):
            batch = {k: [] for k in train_keys}
            for x in batch_list:
                for k in train_keys:
                    batch[k].append(torch.from_numpy(x[k].to_numpy()))
            return {k: torch.stack(v) for k, v in batch.items()}

        return DataLoader(
            self.train_dp,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        val_keys = [
            "kspace_data",
            "kspace_data_cse",
            "kspace_traj",
            "kspace_traj_cse",
            "P2PCSE",
        ]

        def collate_fn(batch_list):
            return_list = []
            for kd, init in batch_list:
                batch = {k: None for k in val_keys}
                for k in val_keys:
                    if k == "P2PCSE":
                        batch[k] = torch.from_numpy(init[k].to_numpy())
                    else:
                        batch[k] = torch.from_numpy(kd[k].to_numpy())
                return_list.append(batch)
            return return_list

        return DataLoader(
            self.val_dp,
            batch_size=self.val_batch_size,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self):
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
    data = BlackBone()
    data.prepare_data()
    data.setup()

# %%
