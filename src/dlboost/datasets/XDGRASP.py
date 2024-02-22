import os
from glob import glob
from pathlib import Path

import dask
import numpy as np
import torch
import xarray as xr
import zarr

# from dataclasses import dataclass
from dlboost.datasets.boilerplate import recon_one_scan
from einops import rearrange
from lightning.pytorch import LightningDataModule
from mrboost import io_utils as iou
from torch.utils.data import DataLoader, Dataset


# %%
class DCE_XDGRASP_KXKYZ_Dataset(Dataset):
    def __init__(
        self, data, transform=None, mode="fit", patch_size_z=5, filename=None
    ) -> None:
        super().__init__()
        self.keys = [
            "kspace_data_z",
            "kspace_data_z_compensated",
            "kspace_traj",
            "kspace_density_compensation",
        ]
        self.data = data
        self.mode = mode
        self.patch_size_z = patch_size_z
        self.id = Path(filename).stem

    def __len__(self) -> int:
        if self.mode == "fit":
            return self.data["kspace_data_z"].shape[2] // self.patch_size_z
        else:
            return len(self.data)

    def __getitem__(self, index: int):
        if self.mode == "fit":
            start_idx = index * self.patch_size_z
            end_idx = (index + 1) * self.patch_size_z
            kspace_data_z_ = self.data["kspace_data_z"][:, :, start_idx:end_idx, ...]
            kspace_data_z_compensated_ = self.data["kspace_data_z_compensated"][
                :, :, start_idx:end_idx, ...
            ]
            kspace_data_z_fixed = rearrange(
                torch.from_numpy(kspace_data_z_[0::2]),
                "ph ch z sp len -> ph ch z (sp len)",
            )
            kspace_data_z_moved = rearrange(
                torch.from_numpy(kspace_data_z_[1::2]),
                "ph ch z sp len -> ph ch z (sp len)",
            )
            """
            kspace_data_z_compensated_fixed = rearrange(
                torch.from_numpy(kspace_data_z_compensated_[0::2]), 'ph ch z sp len -> ph ch z (sp len)')
            kspace_data_z_compensated_moved = rearrange(
                torch.from_numpy(kspace_data_z_compensated_[1::2]), 'ph ch z sp len -> ph ch z (sp len)')
            kspace_data_z_cse_fixed = rearrange(
                torch.tensor(kspace_data_z_[0::2,..., 240:400]), 'ph ch z sp len -> () ch z (ph sp len)')
            kspace_data_z_cse_moved = rearrange(
                torch.tensor(kspace_data_z_[1::2,..., 240:400]), 'ph ch z sp len -> () ch z (ph sp len)')
            kspace_data_z_cse = torch.cat((kspace_data_z_cse_fixed, kspace_data_z_cse_moved), dim = 0)
            """

            kspace_traj_ = torch.from_numpy(self.data["kspace_traj"][:, :, 0])
            kspace_traj_ = torch.view_as_real(kspace_traj_).to(torch.float32)
            kspace_traj_fixed = rearrange(
                kspace_traj_[0::2], "ph () sp len c -> ph c (sp len)"
            )
            kspace_traj_moved = rearrange(
                kspace_traj_[1::2], "ph () sp len c -> ph c (sp len)"
            )

            print(self.data.keys())
            # kspace_traj_cse_fixed = rearrange(kspace_traj_[0::2,:,:, 240:400], 'ph () sp len c -> () c (ph sp len)')
            # kspace_traj_cse_moved = rearrange(kspace_traj_[1::2,:,:, 240:400], 'ph () sp len c -> () c (ph sp len)')
            # kspace_traj_cse = torch.cat((kspace_traj_cse_fixed, kspace_traj_cse_moved), dim = 0)
            cse = torch.from_numpy(self.cse[:, start_idx:end_idx])  # 15,80,320,320
            return dict(
                kspace_data_z_fixed=kspace_data_z_fixed,
                kspace_data_z_moved=kspace_data_z_moved,
                kspace_traj_fixed=kspace_traj_fixed,
                kspace_traj_moved=kspace_traj_moved,
                cse=cse,
                # kspace_data_z_compensated_fixed=kspace_data_z_compensated_fixed,
                # kspace_data_z_compensated_moved=kspace_data_z_compensated_moved,
                # kspace_data_z_cse_fixed = kspace_data_z_cse_fixed,
                # kspace_data_z_cse_moved = kspace_data_z_cse_moved,
                # kspace_traj_cse_fixed = kspace_traj_cse_fixed,
                # kspace_traj_cse_moved = kspace_traj_cse_moved,
            )
        else:
            kspace_data_z_ = self.data[index]["kspace_data_z"][:3, :, :, 40:41]
            kspace_data_z_compensated_ = self.data[index]["kspace_data_z_compensated"][
                :3, :, :, 40:41
            ]
            kspace_traj_ = torch.from_numpy(
                self.data[index]["kspace_traj"][:3, :, :, 0]
            )  # t, ph, ch=1, z=1, sp, len
            # kspace_data_z_ = self.data[index]["kspace_data_z"][:, :, :, 40:41]
            # kspace_data_z_compensated_ = self.data[index]["kspace_data_z_compensated"][:, :, :, 40:41]
            # kspace_traj_ = torch.from_numpy(
            #     self.data[index]["kspace_traj"][:, :, :, 0]
            # )  #t, ph, ch=1, z=1, sp, len
            kspace_data_z = rearrange(
                torch.from_numpy(kspace_data_z_),
                "t ph ch z sp len -> t ph ch z (sp len)",
            )
            kspace_data_z_compensated = rearrange(
                torch.from_numpy(kspace_data_z_compensated_),
                "t ph ch z sp len -> t ph ch z (sp len)",
            )
            kspace_traj_ = torch.view_as_real(kspace_traj_).to(torch.float32)
            kspace_traj = rearrange(kspace_traj_, "t ph () sp len c -> t ph c (sp len)")
            cse = torch.from_numpy(self.data[index]["cse"][:])[:, 40:41]
            return dict(
                kspace_data_z=kspace_data_z,
                kspace_data_z_compensated=kspace_data_z_compensated,
                # kspace_data_z_cse = kspace_data_z_cse,
                # kspace_traj_cse = kspace_traj_cse,
                kspace_traj=kspace_traj,
                cse=cse,
                id=self.id,
            )

    def __getitems__(self, indices):
        return [self.__getitem__(index) for index in indices]


# %%
class DCE_XDGRASP_KXKYZ(LightningDataModule):
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
        self.patch_size = patch_size
        self.patch_sample_number = patch_sample_number

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.val_batch_size = eval_batch_size
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) / ".DCE_MOTIF_CORD"
        self.num_workers = num_workers
        self.rand_idx = rand_idx
        self.n_splits = n_splits
        print(fold_idx)
        self.fold_idx = fold_idx

        self.dat_file_path_list = dat_file_path_list
        self.patient_ids = [
            (self.data_dir / p).parent.name for p in self.dat_file_path_list
        ]

        # self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        # self.train_idx, self.val_idx = [
        #     (train, test) for train, test in self.kf.split(self.dat_file_path_list)
        # ][self.fold_idx]
        self.val_patient_ids = self.patient_ids

    def prepare_data(self):
        return
        # chunks_d = dict()
        # for k in self.val_keys:
        #     if k == "cse":  # ch,z,h,w
        #         chunks_d[k] = (None, 1, None, None)
        #     elif k == "kspace_traj":  # t,ph,sp,lens
        #         chunks_d[k] = (1, None, None, None)
        #     else:  # t,ph,ch,z,sp,lens
        #         chunks_d[k] = (1, None, None, 1, None, None)

        # for idx, p in enumerate(self.dat_file_path_list[self.val_scope]):
        #     dat_file_to_recon = Path(self.data_dir) / p
        #     patient_id = dat_file_to_recon.parent.name
        #     filename = patient_id + ".zarr"
        #     save_path = self.cache_dir / filename
        #     print(patient_id)
        #     if not iou.check_mk_dirs(save_path):
        #         raw_data = recon_one_scan(
        #             dat_file_to_recon, phase_num=5, time_per_contrast=10
        #         )
        #         t, ph, ch, z, sp, lens = raw_data["kspace_data_z"].shape

        #         dict_data = {
        #             k: zarr.array(raw_data[k].numpy(), chunks=chunks_d[k])
        #             for k in self.val_keys
        #         }
        #         zarr.save(save_path, **dict_data)
        #         print(filename)

    def setup(self, init=False, stage: str = "fit"):
        dask.config.set(scheduler="synchronous")
        val_ds_list = [
            (
                str(self.cache_dir / "val" / f"{pid}.zarr"),
                str((self.cache_dir / "val" / f"{pid}_P2PCSE.zarr")),
            )
            for pid in self.patient_ids[0:1]
        ]
        self.val_dp = [
            (
                xr.open_zarr(ds).isel(t=slice(0, 1), z=slice(40, 41)),
                xr.open_zarr(init).isel(t=slice(0, 1), z=slice(40, 41)),
            )
            for ds, init in val_ds_list
        ]

        # data_filenames = glob(str(self.cache_dir / "ONC-DCE-004.zarr"))
        # data = [zarr.open(filename, mode="r") for filename in data_filenames]
        # self.val_dataset = DCE_XDGRASP_KXKYZ_Dataset(
        #     data, mode="val", filename=data_filenames[0]
        # )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=0,
            shuffle=True,
        )

    def val_dataloader(self):
        val_keys = [
            "kspace_data",
            "kspace_data_compensated",
            # "kspace_data_cse",
            "kspace_traj",
            # "kspace_traj_cse",
            "cse",
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

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.eval_batch_size, num_workers=0
        )


# %%
if __name__ == "__main__":
    # dataset = LibriSpeech()
    # dataset.prepare_data()

    data = DCE_XDGRASP_KXKYZ()
    data.prepare_data()
    data.setup()
