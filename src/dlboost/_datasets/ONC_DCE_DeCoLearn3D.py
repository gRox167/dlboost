# %%
import random
from monai.transforms import Lambdad,EnsureChannelFirstd ,RandGridPatchd, RandSpatialCropSamplesd, Transform, MapTransform, Compose, ToTensord, AddChanneld
from mrboost import io_utils as iou
import h5py
import numpy as np
from monai.transforms.adaptors import adaptor
from monai.data import PatchIterd, Dataset, GridPatchDataset
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from monai.data import DataLoader
from itertools import combinations
import os
import torch
from pathlib import Path
from itertools import combinations, product
from juliacall import Main as jl
jl.include(
    "/data-local/anlab/Chunxu/DL_MOTIF/3_MVF/DeCoLearn3D/datasets/decolearn.jl")
jl.seval("using Glob")

# %%
class ONC_DCE_H5_Read_Transform(MapTransform):
    def __call__(self, item_dict):
        output_dict = dict()
        p = item_dict[self.keys[0]]
        top_channels = jl.check_top_k_channel(str(p), 5)
        ch = random.choice(top_channels)-1
        for key in self.keys:
            # p,ch = item_dict[key]
            p = item_dict[key]
            d = jl.get_data_from_h5(str(p),ch)
            for k, v in d.items():
                output_dict[k+"_"+key] = v.to_numpy()
        # print(output_dict.keys())
        return output_dict
    
# class ONC_DCE_H5_Read_Transform(MapTransform):
#     def __call__(self, item_dict):
#         output_dict = dict()
#         for key in self.keys:
#             p,ch = item_dict[key]
#             with h5py.File(p, 'r') as f:
#                 d = dict( 
#                     kspace_data = f["kspace_data_real"][ch,:,:,:]+f["kspace_data_imag"][ch,:,:,:]*1j,
#                     kspace_density_compensation = f["kspace_density_compensation"][:,:],
#                     kspace_traj = f["kspace_traj_real"][:,:] + f["kspace_traj_imag"][:,:] * 1j,
#                     image = f["multi_ch_img_real"][ch,:,:,:] + f["multi_ch_img_imag"][ch,:,:,:] * 1j,
#                     kspace_mask = f["kspace_data_mask"][ch,:,:,:]
#                 )
#             for k, v in d.items():
#                 output_dict[k+"_"+key] = v
#         return output_dict

# %%
class ONC_DCE_DeCoLearn3D(LightningDataModule):
    def __init__(
        self,
        data_dir: os.PathLike = '/data-local/anlab/Chunxu/DL_MOTIF/recon_results/CCIR_01168_ONC-DCE',
        patch_size = (20,320,320),
        train_batch_size: int = 4,
        eval_batch_size: int = 32,
        num_workers: int = 0,
        # cache_dir: os.PathLike = '/bmr207/nmrgrp/nmr201/.cache',
    ):
        super().__init__()
        self.patch_size = patch_size
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / '.cache'
        self.num_workers = num_workers
        self.contrast, self.phase = 34, 5
        iou.check_mk_dirs(self.cache_dir)

    def prepare_data(self):
        # single thread, download can be done here
        pass

    def generate_data_dict(self, data_path_list):
        data=[]
        for raw_data_path in data_path_list:
            # iter_t_ph = product(range(t),combinations(range(ph), 2))
            iter_t_ph = product(range(self.contrast),[(0,1),(1,2),(2,3),(3,4)])
            for t, pair in iter_t_ph:
                # with h5py.File(Path(raw_data_path)/f"contrast_{t}_phase_{pair[0]}.h5", 'r') as f:
                    # ch = f['kspace_data_real'].shape[0]
                # for i in range(ch):
                item_dict = dict()
                item_dict["fixed"] = Path(raw_data_path)/f"contrast_{t}_phase_{pair[0]}.h5"
                item_dict["moved"] = Path(raw_data_path)/f"contrast_{t}_phase_{pair[1]}.h5"
                    # item_dict["fixed"] = (Path(raw_data_path)/f"contrast_{t}_phase_{pair[0]}.h5",i)
                    # item_dict["moved"] = (Path(raw_data_path)/f"contrast_{t}_phase_{pair[1]}.h5",i)
                data.append(item_dict)
        return data

    def setup(self, init=False, stage: str = 'train'):
        raw_data_list = jl.glob("ONC-DCE-*", str(self.data_dir))
        raw_data_train = raw_data_list[0:1]
        raw_data_val = raw_data_list[0:1]
        raw_data_test = raw_data_list[0:1]
        t,ph = self.contrast, self.phase
   
        train_transforms_before_patchify = Compose([
            ONC_DCE_H5_Read_Transform(keys=["fixed", "moved"]),
            EnsureChannelFirstd(keys=[
                'image_fixed', 'image_moved',], channel_dim="no_channel"),
            ToTensord(device=torch.device('cpu'), keys=[
                'kspace_density_compensation_fixed', 'image_fixed', 'kspace_data_fixed', 'kspace_mask_fixed', 'kspace_traj_fixed',# 'cse_fixed', 
                'kspace_density_compensation_moved', 'image_moved', 'kspace_data_moved', 'kspace_mask_moved', 'kspace_traj_moved',# 'cse_moved'
            ]),
            Lambdad(keys=['image_fixed', 'image_moved'], func=lambda x: x.clone(), overwrite=["image_recon_fixed_cache","image_recon_moved_cache"]),
            RandGridPatchd(keys=['image_fixed',  'image_moved'], patch_size=self.patch_size, overlap=0.25, pad_mode=None)
        ])
        # random.shuffle(data)
        self.train_dataset = Dataset(self.generate_data_dict(raw_data_train), transform=train_transforms_before_patchify)
        eval_transforms_before_patchify = Compose([
                ONC_DCE_H5_Read_Transform(keys=["fixed"]),
                EnsureChannelFirstd(keys=[
                'image_fixed'], channel_dim="no_channel"),
                ToTensord(device=torch.device('cpu'), keys=['image_fixed']),
        ])

  
        self.val_dataset = Dataset(self.generate_data_dict(raw_data_val), transform=eval_transforms_before_patchify)

        data = []
        for raw_data_path in raw_data_test:
            for t, ph in zip(range(t),range(ph)):
                item_dict = dict()
                item_dict["fixed"] = (raw_data_path,t,ph)
                data.append(item_dict)
        self.test_dataset = Dataset(data, transform=eval_transforms_before_patchify)

    def train_dataloader(self):
        # return self.train_dataset
        return self.train_dataset

    def val_dataloader(self):
        return self.val_dataset
        # return DataLoader(self.val_dataset, batch_size=1, collate_fn=self.transfer_batch_to_devic)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, collate_fn=self.transfer_batch_to_device,)

if __name__ == "__main__":
    # dataset = LibriSpeech()
    # dataset.prepare_data()

    data = ONC_DCE_DeCoLearn3D()
    data.prepare_data()
    data.setup()
    for i in data.train_dataloader():
        print(i.keys())
# %%
