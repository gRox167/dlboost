# %%
import random
from itertools import combinations, product
from pathlib import Path
from glob import glob
import os

import zarr
import numpy as np
import torch
# from torch.utils.data import 
from einops import rearrange, reduce, repeat
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from monai.transforms import SplitDimd,Lambdad, EnsureChannelFirstd, RandGridPatchd, RandSpatialCropSamplesd, Transform, MapTransform, Compose, ToTensord, AddChanneld
from monai.transforms.adaptors import adaptor
from monai.data import PatchIterd, Dataset, PatchDataset
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
 

class ONC_DCE_P2P(LightningDataModule):
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
        self.cache_dir = self.data_dir / '.cache'
        self.num_workers = num_workers
        self.phase = 10
        iou.check_mk_dirs(self.cache_dir)

    def prepare_data(self):
        # single thread, download can be done here
        pass

    def generate_data_dict(self, data_path_list, train_flag = True):
        data_list = []
        for p in data_path_list:
            phase_list = [zarr.open(self.data_dir/p/f"phase_{ph}"/"img_abs.zarr", mode='r') for ph in range(self.phase)]
            data_list.append(phase_list) # load all phases path as dict
        if train_flag:
            data = np.concatenate(data_list, axis=0)
            img_fixed, img_moved = data[:,0::2], data[:,1::2]
            data = [dict(img_fixed=img_fixed[i], img_moved=img_moved[i]) for i in range(img_fixed.shape[0])]
        else:
            data = [dict(img=data_list[i]) for i in range(len(data_list))]
        return data

    def setup(self, init=False, stage: str = 'train'):
        raw_data_list = glob("ONC-DCE-*", root_dir = self.data_dir)
        raw_data_train = raw_data_list[0:6]
        raw_data_val = raw_data_list[6:7]
        raw_data_test = raw_data_list[7:8]
        if stage == 'train' or stage is None:  
            train_transforms = Compose([
                EnsureChannelFirstd(keys=[
                    'img_fixed', 'img_moved'], channel_dim="no_channel"),
                ToTensord(device=torch.device('cpu'), keys=['img_fixed', 'img_moved']),
            ])
            self.train_dataset = Dataset(self.generate_data_dict(raw_data_train), train_transforms)
        eval_transforms = Compose([
                EnsureChannelFirstd(keys=[
                    'img'], channel_dim="no_channel"),
                ToTensord(device=torch.device('cpu'), keys=['img_fixed', 'img_moved'])])
        self.val_dataset = Dataset(self.generate_data_dict(
            raw_data_val, train_flag=False), transform=eval_transforms)
        self.test_dataset = Dataset(self.generate_data_dict(
            raw_data_test), transform=eval_transforms)

    def train_dataloader(self):
        # return self.train_dataset
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, num_workers=0)
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
