# %%
import os
import random
from glob import glob
from pathlib import Path

import dask
import torch
import xarray as xr

# from einops import rearrange
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

# from dataclasses import dataclass
from dlboost.datasets.boilerplate import recon_one_scan


class P2PCSE_TRAIN(Dataset):
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
        sample = xr.open_zarr(self.train_kd[0])
        t_size = sample.sizes["t"]
        t_indices = [i for i in range(t_size)] * self.patch_sample_number
        # z axis have z slices, we want to randomly sample n patches from these slices, each patch have p slices.

        z_slices = [
            slice(start_idx, self.patch_size[0] + start_idx)
            for start_idx in random.choices(
                range(sample.sizes["z"] - self.patch_size[0]), k=len(t_indices)
            )
        ]
        self.train_idx = [
            (kd, t_idx, z_slice)
            for kd in self.train_kd
            for t_idx, z_slice in zip(t_indices, z_slices)
        ]

    def __len__(self):
        return len(self.train_idx)

    def __getitem__(self, index):
        def get_data_dict(train_kd, t_idx, z_slice):
            return {
                "kspace_data_odd": xr.open_zarr(train_kd)["kspace_data_odd"].isel(
                    t=t_idx, z=z_slice
                ),
                "kspace_data_even": xr.open_zarr(train_kd)["kspace_data_even"].isel(
                    t=t_idx, z=z_slice
                ),
                # "kspace_data_compensated_odd": xr.open_zarr(train_kd)[
                #     "kspace_data_compensated_odd"
                # ].isel(t=t_idx, z=z_slice),
                # "kspace_data_compensated_even": xr.open_zarr(train_kd)[
                #     "kspace_data_compensated_even"
                # ].isel(t=t_idx, z=z_slice),
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
            }

        return get_data_dict(*self.train_idx[index])


class P2PCSE_VAL(Dataset):
    def __init__(
        self, cache_dir, patient_ids, t_slice=slice(None), z_slice=slice(None)
    ):
        self.cache_dir = cache_dir
        self.patient_ids = patient_ids
        self.val_kd = [
            str(self.cache_dir / "val" / f"{pid}.zarr") for pid in self.patient_ids
        ]
        self.t_slice = t_slice
        self.z_slice = z_slice

    def __len__(self):
        return len(self.val_kd)

    def __getitem__(self, index):
        return xr.open_zarr(self.val_kd[index]).isel(t=self.t_slice, z=self.z_slice)


class P2PCSE_Predict(Dataset):
    def __init__(
        self, cache_dir, patient_ids, t_slice=slice(None), z_slice=slice(None)
    ):
        self.cache_dir = cache_dir
        self.patient_ids = patient_ids
        self.train = [
            str(self.cache_dir / "train" / f"{pid}.zarr") for pid in self.patient_ids
        ]
        self.t_slice = t_slice
        self.z_slice = z_slice

    def __len__(self):
        return len(self.train)

    def __getitem__(self, index):
        return xr.open_zarr(self.train[index]).isel(t=self.t_slice, z=self.z_slice)


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
            (self.data_dir / p).parent.name + "_DB"
            if "_Dyn_DB.dat" in p
            else (self.data_dir / p).parent.name
            for p in self.dat_file_path_list
        ]
        print(self.patient_ids, len(self.patient_ids))

        # self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        # self.train_idx, self.val_idx = [
        #     (train, test) for train, test in self.kf.split(self.dat_file_path_list)
        # ][self.fold_idx]

    def prepare_data(self):
        # self.train_save_path = self.cache_dir / str(self.fold_idx) / "train"
        self.train_save_path = self.cache_dir / "train"
        train_integrate = False

        if os.path.exists(self.train_save_path):
            if len(glob(str(self.train_save_path / "*.zarr"))) == len(self.patient_ids):
                train_integrate = True
        if not train_integrate:
            # dask.config.set(scheduler="synchronous")
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
            # zarr_path = val_save_path / (patient_id + "_DB.zarr")
            zarr_path = val_save_path / (patient_id + ".zarr")
            if os.path.exists(zarr_path):
                continue
            raw_data = recon_one_scan(
                dat_file_to_recon, phase_num=5, time_per_contrast=10
            )
            ds = xr.Dataset(
                data_vars=dict(
                    kspace_data=xr.DataArray(
                        raw_data["kspace_data_z"].numpy(),
                        dims=["t", "ph", "ch", "z", "sp", "lens"],
                    ),
                    kspace_data_compensated=xr.DataArray(
                        raw_data["kspace_data_z_compensated"].numpy(),
                        dims=["t", "ph", "ch", "z", "sp", "lens"],
                    ),
                    kspace_data_cse=xr.DataArray(
                        raw_data["kspace_data_z"][..., 240:400].numpy(),
                        dims=["t", "ph2", "ch", "z", "sp2", "lens2"],
                    ),
                    kspace_traj=xr.DataArray(
                        raw_data["kspace_traj"].numpy(),
                        dims=["t", "ph", "complex", "sp", "lens"],
                    ),
                    kspace_traj_cse=xr.DataArray(
                        raw_data["kspace_traj"][..., 240:400].numpy(),
                        dims=["t", "ph2", "complex", "sp2", "lens2"],
                    ),
                    cse=xr.DataArray(
                        raw_data["cse"].numpy(),
                        dims=["ch", "z", "h", "w"],
                    ),
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
            ds.to_zarr(zarr_path)

    def generate_train_dataset(self, train_save_path):
        # iou.check_mk_dirs(self.cache_dir / str(self.fold_idx))
        # for p, patient_id in zip(self.dat_file_path_list, self.patient_ids):
        for p, patient_id in zip(self.dat_file_path_list, self.patient_ids):
            dat_file_to_recon = Path(self.data_dir) / p
            zarr_path = train_save_path / (patient_id + ".zarr")
            if os.path.exists(zarr_path):
                continue
            raw_data = recon_one_scan(
                dat_file_to_recon, phase_num=10, time_per_contrast=20
            )
            ds = xr.Dataset(
                data_vars=dict(
                    kspace_data_odd=xr.DataArray(
                        raw_data["kspace_data_z"][:, 0::2].numpy(),
                        dims=["t", "ph", "ch", "z", "sp", "lens"],
                    ),
                    kspace_data_even=xr.DataArray(
                        raw_data["kspace_data_z"][:, 1::2].numpy(),
                        dims=["t", "ph", "ch", "z", "sp", "lens"],
                    ),
                    kspace_data_compensated_odd=xr.DataArray(
                        raw_data["kspace_data_z_compensated"][:, 0::2].numpy(),
                        dims=["t", "ph", "ch", "z", "sp", "lens"],
                    ),
                    kspace_data_compensated_even=xr.DataArray(
                        raw_data["kspace_data_z_compensated"][:, 1::2].numpy(),
                        dims=["t", "ph", "ch", "z", "sp", "lens"],
                    ),
                    kspace_data_cse_odd=xr.DataArray(
                        raw_data["kspace_data_z"][:, 0::2, ..., 240:400].numpy(),
                        dims=["t", "ph2", "ch", "z", "sp2", "lens2"],
                    ),
                    kspace_data_cse_even=xr.DataArray(
                        raw_data["kspace_data_z"][:, 1::2, ..., 240:400].numpy(),
                        dims=["t", "ph2", "ch", "z", "sp2", "lens2"],
                    ),
                    kspace_traj_odd=xr.DataArray(
                        raw_data["kspace_traj"][:, 0::2].numpy(),
                        dims=["t", "ph", "complex", "sp", "lens"],
                    ),
                    kspace_traj_even=xr.DataArray(
                        raw_data["kspace_traj"][:, 1::2].numpy(),
                        dims=["t", "ph", "complex", "sp", "lens"],
                    ),
                    kspace_traj_cse_odd=xr.DataArray(
                        raw_data["kspace_traj"][:, 0::2, ..., 240:400].numpy(),
                        dims=["t", "ph2", "complex", "sp2", "lens2"],
                    ),
                    kspace_traj_cse_even=xr.DataArray(
                        raw_data["kspace_traj"][:, 1::2, ..., 240:400].numpy(),
                        dims=["t", "ph2", "complex", "sp2", "lens2"],
                    ),
                    cse=xr.DataArray(
                        raw_data["cse"].numpy(),
                        dims=["ch", "z", "h", "w"],
                    ),
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
            ds.to_zarr(zarr_path)

    def setup(self, init=False, stage: str = "fit"):
        if stage == "fit":
            dask.config.set(scheduler="synchronous")
            self.train_ds = P2PCSE_TRAIN(
                self.cache_dir,
                self.patient_ids,
                self.patch_size,
                self.patch_sample_number,
            )
        self.val_ds = P2PCSE_VAL(
            self.cache_dir, self.patient_ids[0:1], slice(0, 4), slice(32, 33)
        )
        self.pred_ds = P2PCSE_Predict(
            self.cache_dir, self.patient_ids, slice(0, 3), slice(30, 50)
        )
        self.test_ds = P2PCSE_VAL(
            self.cache_dir, self.patient_ids, slice(0, 1), slice(30, 50)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=False,
            collate_fn=lambda batch_list: [
                {k: torch.from_numpy(v.to_numpy()) for k, v in x.items()}
                for x in batch_list
            ],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            num_workers=0,
            pin_memory=False,
            collate_fn=lambda x: x,
            # collate_fn=lambda batch_list: [
            #     {k: torch.from_numpy(v.to_numpy()) for k, v in x.data_vars.items()}
            #     for x in batch_list
            # ],
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_ds,
            batch_size=self.val_batch_size,
            num_workers=0,
            pin_memory=False,
            collate_fn=lambda x: x,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.val_batch_size,
            num_workers=0,
            pin_memory=False,
            collate_fn=lambda x: x,
        )

    def transfer_batch_to_device(
        self, batch, device: torch.device, dataloader_idx: int
    ):
        print(self.trainer.state)
        if self.trainer.state == "fit":
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
