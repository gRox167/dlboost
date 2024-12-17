# %%
import os
from glob import glob
from pathlib import Path

import torch
import zarr
from einops import rearrange
from mrboost import io_utils as iou
from pytorch_lightning import (LightningDataModule)
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

from dlboost.datasets.boilerplate import recon_one_scan

@dataclass
class MRReconInput:
    data: torch.Tensor
    image: torch.Tensor
    csm: torch.Tensor
    mvf: torch.Tensor
    omega: torch.Tensor
    w: torch.Tensor

    def clone(self):
        return MRReconInput(
            data=self.data.clone(),
            image=self.image.clone(),
            csm=self.csm.clone(),
            mvf=self.mvf.clone(),
            omega=self.omega.clone(),
            w=self.w.clone()
        )
        
    def to(self, device):
        return MRReconInput(
            data=self.data.to(device),
            image=self.image.to(device),
            csm=self.csm.to(device),
            mvf=self.mvf.to(device),
            omega=self.omega.to(device),
            w=self.w.to(device)
        )

# %%
class DCE_MOTIF_KXKYZ_Dataset(Dataset):
    def __init__(self, data, transform=None, mode = "train", patch_size_z = 5, filename=None) -> None:
        super().__init__()
        self.keys = ["kspace_data_z", "kspace_data_z_compensated",
                     "kspace_traj", "kspace_density_compensation"]
        self.data = data
        self.mode = mode
        self.patch_size_z = patch_size_z
        self.id = Path(filename).stem

    def __len__(self) -> int:
        if self.mode == "train":
            return self.data["kspace_data_z"].shape[2]//self.patch_size_z
        else:
            return len(self.data)

    def __getitem__(self, index: int):
        if self.mode == "train":
            start_idx = index*self.patch_size_z
            end_idx = (index+1)*self.patch_size_z
            kspace_data_z_ = self.data["kspace_data_z"][:, :, start_idx: end_idx, ...]
            kspace_data_z_compensated_ = self.data["kspace_data_z_compensated"][:,:, start_idx: end_idx,...]
            kspace_data_z_fixed = rearrange(
                torch.from_numpy(kspace_data_z_[0::2]), 'ph ch z sp len -> ph ch z (sp len)')
            kspace_data_z_moved = rearrange(
                torch.from_numpy(kspace_data_z_[1::2]), 'ph ch z sp len -> ph ch z (sp len)')
            kspace_data_z_compensated_fixed = rearrange(
                torch.from_numpy(kspace_data_z_compensated_[0::2]), 'ph ch z sp len -> ph ch z (sp len)')
            kspace_data_z_compensated_moved = rearrange(
                torch.from_numpy(kspace_data_z_compensated_[1::2]), 'ph ch z sp len -> ph ch z (sp len)')
            kspace_data_z_cse_fixed = rearrange(
                torch.tensor(kspace_data_z_[0::2,..., 240:400]), 'ph ch z sp len -> () ch z (ph sp len)')
            kspace_data_z_cse_moved = rearrange(
                torch.tensor(kspace_data_z_[1::2,..., 240:400]), 'ph ch z sp len -> () ch z (ph sp len)')
            kspace_data_z_cse = torch.cat((kspace_data_z_cse_fixed, kspace_data_z_cse_moved), dim = 0)

            kspace_traj_ = torch.from_numpy(self.data["kspace_traj"][:, :, 0])
            kspace_traj_ = torch.view_as_real(kspace_traj_).to(torch.float32)
            kspace_traj_fixed = rearrange(kspace_traj_[0::2], 'ph () sp len c -> ph c (sp len)')
            kspace_traj_moved = rearrange(kspace_traj_[1::2], 'ph () sp len c -> ph c (sp len)')
            
            kspace_traj_cse_fixed = rearrange(kspace_traj_[0::2,:,:, 240:400], 'ph () sp len c -> () c (ph sp len)')
            kspace_traj_cse_moved = rearrange(kspace_traj_[1::2,:,:, 240:400], 'ph () sp len c -> () c (ph sp len)')
            # kspace_traj_cse = torch.cat((kspace_traj_cse_fixed, kspace_traj_cse_moved), dim = 0)

            return dict(
                kspace_data_z_fixed=kspace_data_z_fixed,
                kspace_data_z_moved=kspace_data_z_moved,
                kspace_data_z_compensated_fixed=kspace_data_z_compensated_fixed,
                kspace_data_z_compensated_moved=kspace_data_z_compensated_moved,
                kspace_traj_fixed=kspace_traj_fixed,
                kspace_traj_moved=kspace_traj_moved,
                kspace_data_z_cse_fixed = kspace_data_z_cse_fixed,
                kspace_data_z_cse_moved = kspace_data_z_cse_moved,
                kspace_traj_cse_fixed = kspace_traj_cse_fixed,
                kspace_traj_cse_moved = kspace_traj_cse_moved,
                )
        else:
            kspace_data_z_ = self.data[index]["kspace_data_z"][:]
            kspace_data_z_compensated_ = self.data[index]["kspace_data_z_compensated"][:]
            kspace_traj_ = torch.from_numpy(self.data[index]["kspace_traj"][:, :, 0]) # ph, ch=1, z=1, sp, len
            kspace_data_z = rearrange(
                torch.from_numpy(kspace_data_z_), 'ph ch z sp len -> ph ch z (sp len)')
            kspace_data_z_compensated = rearrange(
                torch.from_numpy(kspace_data_z_compensated_), 'ph ch z sp len -> ph ch z (sp len)')
            kspace_data_z_cse = rearrange(
                torch.tensor(kspace_data_z_[..., 240:400]), 'ph ch z sp len -> () ch z (ph sp len)')
            kspace_traj_ = torch.view_as_real(kspace_traj_).to(torch.float32)
            kspace_traj = rearrange(kspace_traj_, 'ph () sp len c -> ph c (sp len)')
            
            kspace_traj_cse = rearrange(kspace_traj_[:,:,:,240:400], 'ph () sp len c -> () c (ph sp len)')
            return dict(
                kspace_data_z=kspace_data_z,
                kspace_data_z_compensated=kspace_data_z_compensated,
                kspace_data_z_cse = kspace_data_z_cse,
                kspace_traj_cse = kspace_traj_cse,
                kspace_traj=kspace_traj,
                id = self.id)
            

    def __getitems__(self, indices):
        return [self.__getitem__(index) for index in indices]




# %%
class DCE_MOTIF_KXKYZ(LightningDataModule):
    def __init__(
        self,
        data_dir: os.PathLike = '/data/anlab/RawData_MR/',
        cache_dir: os.PathLike = Path("/data-local/anlab/Chunxu/.cache"),
        train_scope=slice(0, 6),
        val_scope=slice(6, 9),
        load_scope=slice(0, -1),
        patch_size=(20, 320, 320),
        num_samples_per_subject=16,
        train_batch_size: int = 4,
        eval_batch_size: int = 1,
        num_workers: int = 0,
    ):
        super().__init__()
        self.train_scope = train_scope
        self.val_scope = val_scope
        self.load_scope = load_scope
        self.patch_size = patch_size
        self.num_samples_per_subject = num_samples_per_subject
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.data_dir = Path(data_dir)
        self.cache_dir = cache_dir / '.DCE_P2PCSE_KXKY_10ph'
        self.num_workers = num_workers
        self.contrast, self.phase = 34, 5
        self.keys = ["kspace_data_z", "kspace_data_z_compensated",
                     "kspace_traj", "kspace_density_compensation"]
        self.val_keys = ["kspace_data_z",
                         "kspace_data_z_compensated", "kspace_traj", "cse"]
        # self.raw_data_list = glob("ONC-DCE-*", root_dir = self.data_dir)
        # self.top_k=5

        self.dat_file_path_list = [
            "CCIR_01168_ONC-DCE/ONC-DCE-003/meas_MID00781_FID11107_CAPTURE_FA15_Dyn.dat",
            "CCIR_01168_ONC-DCE/ONC-DCE-005/meas_MID01282_FID10023_CAPTURE_FA15_Dyn.dat",
            "CCIR_01168_ONC-DCE/ONC-DCE-006/meas_MID00221_FID07916_Abd_CAPTURE_FA13_Dyn.dat",
            "CCIR_01168_ONC-DCE/ONC-DCE-007/meas_MID00106_FID17478_CAPTURE_FA15_Dyn.dat",
            "CCIR_01168_ONC-DCE/ONC-DCE-008/meas_MID00111_FID14538_CAPTURE_FA15_Dyn.dat",
            "CCIR_01168_ONC-DCE/ONC-DCE-009/meas_MID00319_FID19874_CAPTURE_FA15_Dyn.dat",
            "CCIR_01168_ONC-DCE/ONC-DCE-010/meas_MID00091_FID19991_CAPTURE_FA13_Dyn.dat",
            "CCIR_01168_ONC-DCE/ONC-DCE-011/meas_MID00062_FID07015_CAPTURE_FA15_Dyn.dat",
            "CCIR_01168_ONC-DCE/ONC-DCE-012/meas_MID00124_FID07996_CAPTURE_FA14_Dyn.dat",
            "CCIR_01168_ONC-DCE/ONC-DCE-013/meas_MID00213_FID10842_CAPTURE_FA15_Dyn.dat",
            "CCIR_01168_ONC-DCE/ONC-DCE-014/meas_MID00099_FID12331_CAPTURE_FA14_5_Dyn.dat",
            "CCIR_01135_NO-DCE/NO-DCE-002/meas_MID00042_FID44015_CAPTURE_FA15_Dyn.dat",
            "CCIR_01135_NO-DCE/NO-DCE-003/meas_MID01259_FID07773_CAPTURE_FA15_Dyn.dat",
            "CCIR_01135_NO-DCE/NO-DCE-004/meas_MID02372_FID14845_CAPTURE_FA15_Dyn.dat",
            "CCIR_01135_NO-DCE/NO-DCE-005/meas_MID00259_FID01679_CAPTURE_FA15_Dyn.dat",
            "CCIR_01135_NO-DCE/NO-DCE-006/meas_MID01343_FID04307_CAPTURE_FA15_Dyn.dat",
            "CCIR_01135_NO-DCE/NO-DCE-008/meas_MID00888_FID06847_CAPTURE_FA15_Dyn.dat",

            "CCIR_01168_ONC-DCE/ONC-DCE-004/meas_MID00165_FID04589_Abd_CAPTURE_FA15_Dyn.dat",
            "CCIR_01135_NO-DCE/NO-DCE-009/meas_MID00912_FID18265_CAPTURE_FA15_Dyn.dat",
            "CCIR_01135_NO-DCE/NO-DCE-001/meas_MID00869_FID13275_CAPTURE_FA15_Dyn.dat",
            "CCIR_01168_ONC-DCE/ONC-DCE-001/meas_MID00144_FID02406_CAPTURE_FA15_Dyn.dat",
        ]

    def prepare_data(self):
        # single thread, download can be done here
        if not iou.check_mk_dirs(self.cache_dir/"train"):
            for idx, p in enumerate(self.dat_file_path_list[self.train_scope]):
                dat_file_to_recon = Path(self.data_dir)/p
                patient_id = dat_file_to_recon.parent.name
                print(patient_id)
                raw_data = recon_one_scan(
                    dat_file_to_recon, phase_num=10, time_per_contrast=20)
                t, ph, ch, z, sp, lens = raw_data["kspace_data_z"].shape
                # topk_ch = check_top_k_channel(raw_data["kspace_data_z"], k = self.top_k)
                for contra in range(t):
                    dict_data = {k: raw_data[k][contra].numpy()
                                 for k in self.keys}
                    filename = patient_id+"_"+str(contra)+".zarr"
                    save_path = self.cache_dir/"train"/filename
                    if not iou.check_mk_dirs(save_path):
                        zarr.save(save_path, **dict_data)
                        print(filename)
        if not iou.check_mk_dirs(self.cache_dir/"val"):
            for idx, p in enumerate(self.dat_file_path_list[self.val_scope]):
                dat_file_to_recon = Path(self.data_dir)/p
                patient_id = dat_file_to_recon.parent.name
                print(patient_id)
                raw_data = recon_one_scan(
                    dat_file_to_recon, phase_num=5, time_per_contrast=10)
                t, ph, ch, kz, sp, lens = raw_data["kspace_data_z"].shape
                for contra in range(t):
                    dict_data = {k: raw_data[k][contra].numpy()
                                 for k in self.keys}
                    filename = patient_id+"_"+str(contra)+".zarr"
                    save_path = self.cache_dir/"val"/filename
                    if not iou.check_mk_dirs(save_path):
                        zarr.save(save_path, **dict_data)
                        print(filename)

    def setup(self, init=False, stage: str = 'train'):
        if stage == 'train' or stage == 'continue':
            data_filenames = glob(str(self.cache_dir/"train/*.zarr"))[0:1]
            train_datasets = [DCE_MOTIF_KXKYZ_Dataset(
                zarr.open(filename, mode='r'), patch_size_z= self.patch_size[0], filename=filename) for filename in data_filenames]
            self.train_dataset = torch.utils.data.ConcatDataset(train_datasets)

        data_filenames = glob(str(self.cache_dir/"val/ONC-DCE-004_0.zarr"))
        data  = [zarr.open(filename, mode='r') for filename in data_filenames]
        self.val_dataset =DCE_MOTIF_KXKYZ_Dataset(data[0], mode = "val", filename=data_filenames[0])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=0, collate_fn=lambda x: x, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=2, num_workers=0, collate_fn=lambda x: x, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, num_workers=0)


# %%
if __name__ == "__main__":
    # dataset = LibriSpeech()
    # dataset.prepare_data()

    data = DCE_MOTIF_KXKYZ()
    data.prepare_data()
    data.setup()
    # for i in data.train_dataloader():
    #     print(i["kspace_data_z"].device)
# %%
