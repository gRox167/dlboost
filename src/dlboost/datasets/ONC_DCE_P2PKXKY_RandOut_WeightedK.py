# %%
from curses import KEY_BACKSPACE
from ipaddress import collapse_addresses
import random
from itertools import combinations, product
from pathlib import Path
from glob import glob
import os

import zarr
import numpy as np
import torch
from einops import rearrange, reduce, repeat
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from monai.transforms import SplitDimd,Lambdad, EnsureChannelFirstd, RandGridPatchd, RandSpatialCropSamplesd, Transform, MapTransform, Compose, ToTensord, AddChanneld
from monai.transforms.adaptors import adaptor
from monai.data import PatchIterd, Dataset, PatchDataset, IterableDataset
from monai.data import DataLoader

from mrboost import io_utils as iou


class ONC_DCE_Zarr_Read_Transform(MapTransform):
    def __init__(self, keys,train_flag=True):
        super().__init__(keys)
        self.train_flag = train_flag

    def __call__(self, item_dict):
        if self.train_flag:
            output_dict = dict()
            p = item_dict[self.keys[0]]
            for key in self.keys:
                path_list = item_dict[key]
                d = np.concatenate([zarr.open(p, mode='r') for p in path_list], axis=0)
                output_dict["img_abs_"+key] = d
        return output_dict
 

def check_top_k_channel(path, k = 5):
    d = zarr.open(path, mode='r')
    ph, ch, kz, sp, lens = d.shape
    if ch < k:
        return d 
    else:
        center_len  = lens//2
        center_z = kz//2
        lowk_energy = [np.sum(np.abs(d[0,ch,center_z-5:center_z+5, :, center_len-20:center_len+20])) for ch in range(ch)]
        # print(lowk_energy)
        sorted_energy, sorted_idx = torch.sort(torch.tensor(lowk_energy), descending=True)
        # print(sorted_energy, sorted_idx)
        return sorted_idx[:k].tolist()

class ONC_DCE_P2PKXKY(LightningDataModule):
    def __init__(
        self,
        data_dir: os.PathLike = '/data-local/anlab/Chunxu/DL_MOTIF/recon_results_for_P2P/CCIR_01168_ONC-DCE',
        patch_size=(20, 320, 320),
        num_samples_per_subject= 16,
        train_batch_size: int = 4,
        eval_batch_size: int = 1,
        num_workers: int = 0,
        # cache_dir: os.PathLike = '/bmr207/nmrgrp/nmr201/.cache',
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_samples_per_subject = num_samples_per_subject
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / '.cache_ONC_DCE_P2PKXKY_RandOut_WeightedK'
        self.num_workers = num_workers
        self.phase = 10
        # self.keys = ["kspace_data_z_imag.zarr", "kspace_data_z_real.zarr", "kspace_data_z_compensated_imag.zarr", "kspace_data_z_compensated_real.zarr", "kspace_traj_imag.zarr", "kspace_traj_real.zarr"]
        self.keys = ["kspace_data_z", "kspace_data_z_compensated", "kspace_traj", "kspace_density_compensation"]
        self.val_keys = ["kspace_data_z", "kspace_data_z_compensated", "kspace_traj", "cse"]
        self.raw_data_list = glob("ONC-DCE-*", root_dir = self.data_dir)

        iou.check_mk_dirs(self.cache_dir)

    def prepare_data(self):
        self.raw_data_train = self.raw_data_list[0:6]
        self.raw_data_val = self.raw_data_list[6:7]
        if not iou.check_mk_dirs(self.cache_dir/"train"):
            k="kspace_data_z"
            topk_ch_list = [check_top_k_channel(self.data_dir/path/(k+".zarr"), k = 5) for path in self.raw_data_train]
            print(topk_ch_list)
            for k in self.keys:
                _fixed,_moved = [],[]
                for path,topk_ch in zip(self.raw_data_train, topk_ch_list):
                    phase = zarr.open(self.data_dir/path/(k+".zarr"), mode='r')
                    phase = np.array(phase.get_orthogonal_selection((slice(None),topk_ch,slice(None),slice(None),slice(None))))
                    phase_fixed = phase[0::2]
                    phase_moved = phase[1::2]
                    d_fixed = rearrange(phase_fixed, 'ph ch d sp len -> (ch d) ph sp len')
                    d_moved = rearrange(phase_moved, 'ph ch d sp len -> (ch d) ph sp len')
                    _fixed.append(d_fixed)
                    _moved.append(d_moved)
                zarr.save(self.cache_dir/"train"/(k+"_fixed"+".zarr"), np.concatenate(_fixed, axis=0))
                zarr.save(self.cache_dir/"train"/(k+"_moved"+".zarr"), np.concatenate(_moved, axis=0))
        if not iou.check_mk_dirs(self.cache_dir/"val"):
            for k in self.val_keys:
                for path in self.raw_data_val:
                    d = np.array(zarr.open(self.data_dir/path/(k+".zarr"), mode='r'))
                    d = rearrange(d, 'ph ch d sp len  -> ch d ph sp len')
                    zarr.save(self.cache_dir/"val"/(path+"_"+k+".zarr"), d)

    def setup(self, init=False, stage: str = 'train'):
        if stage == 'train' or stage == 'continue':
            train_iter = zip(*[zarr.open(self.cache_dir/"train"/(k+".zarr"), mode='r') for k in ["kspace_data_z_compensated_fixed", "kspace_data_z_fixed", "kspace_traj_fixed", "kspace_density_compensation_fixed", "kspace_data_z_compensated_moved", "kspace_data_z_moved", "kspace_traj_moved", "kspace_density_compensation_moved"]])
            train_data_list = [dict(kspace_data_compensated_fixed = kcf, 
                                kspace_data_fixed = kf,
                                kspace_traj_fixed = ktf,
                                kspace_density_compensation_fixed = kdf,
                                kspace_data_compensated_moved = kcm,
                                kspace_data_moved = km,
                                kspace_traj_moved = ktm,
                                kspace_density_compensation_moved = kdm) for kcf, kf, ktf, kdf, kcm, km, ktm,kdm in train_iter]
            train_transforms = Compose([
                # EnsureChannelFirstd(keys=[
                #     'kspace_data_compensated_fixed', 'kspace_data_fixed',
                #     'kspace_data_compensated_moved', 'kspace_data_moved',], channel_dim="no_channel"),
                ToTensord(device=torch.device('cpu'), keys=[
                    'kspace_data_compensated_fixed', 'kspace_data_fixed', 'kspace_traj_fixed', 'kspace_density_compensation_fixed',
                    'kspace_data_compensated_moved', 'kspace_data_moved', 'kspace_traj_moved', 'kspace_density_compensation_moved',
                ]),
                # Lambdad(keys=['kspace_data_fixed', 'kspace_data_moved'], func=lambda x: x/160),
            ])
            self.train_dataset = Dataset(train_data_list, transform=train_transforms)
        val_data_list = [dict(
                kspace_data_compensated = np.array(zarr.open(self.cache_dir/"val"/( path+"_kspace_data_z_compensated.zarr" ), mode='r')),
                # kspace_data = np.array(zarr.open(self.cache_dir/"val"/(path+"_kspace_data_z.zarr"), mode='r')),
                kspace_traj = np.array(zarr.open(self.cache_dir/"val"/(path+"_kspace_traj.zarr"), mode='r')),
                cse = np.array(zarr.open(self.cache_dir/"val"/(path+"_cse.zarr"), mode='r')),
            ) for path in self.raw_data_val]
        eval_transforms = Compose([
                # EnsureChannelFirstd(keys=[
                #     ''], channel_dim="no_channel"),
                ToTensord(device=torch.device('cpu'), keys=['kspace_data_compensated','kspace_traj', 'cse']),
            ])
        self.val_dataset = Dataset(val_data_list, transform=eval_transforms)
        # self.test_dataset = Dataset(self.generate_data_dict(
        #     raw_data_test), transform=eval_transforms)

    def train_dataloader(self):
        # return self.train_dataset
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, num_workers=0, collate_fn=lambda x: x)
        # return DataLoader(self.val_dataset, batch_size=1, collate_fn=self.transfer_batch_to_devic)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, collate_fn=self.transfer_batch_to_device,)

# %%
if __name__ == "__main__":
    # dataset = LibriSpeech()
    # dataset.prepare_data()

    data = ONC_DCE_P2P()
    data.prepare_data()
    data.setup()
    for i in data.train_dataloader():
        print(i.keys())
        print(i["img_fixed"].shape)
        break

# %%
