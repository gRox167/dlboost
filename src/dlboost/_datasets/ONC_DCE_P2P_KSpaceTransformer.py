# %%
from pathlib import Path
from glob import glob
import os

import zarr
import numpy as np
import torch
from tqdm import tqdm
from einops import rearrange, reduce, repeat
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from monai.transforms import Lambda,SplitDimd,Lambdad, EnsureChannelFirstd, RandGridPatchd, RandSpatialCropSamplesd, Transform, MapTransform, Compose, ToTensord, AddChanneld
from monai.data import PatchIterd, Dataset, PatchDataset, IterableDataset
from monai.data import DataLoader

from mrboost import io_utils as iou
from mrboost import reconstruction as recon

def recon_one_scan(dat_file_to_recon):
    reconstructor = recon.CAPTURE_VarW_NQM_DCE_PostInj(
        dat_file_location=dat_file_to_recon, phase_num=5, which_slice=-1, which_contra=-1, which_phase=-1, device=torch.device('cuda:0'))
    raw_data = reconstructor.get_raw_data(reconstructor.dat_file_location)
    reconstructor.args_init()

    preprocessed_data = reconstructor.data_preprocess(raw_data)
    kspace_data_centralized, kspace_data_mask, kspace_data_z, kspace_traj, kspace_density_compensation, cse =\
        preprocessed_data['kspace_data_centralized'], preprocessed_data['kspace_data_mask'], preprocessed_data['kspace_data_z'], \
        preprocessed_data['kspace_traj'], preprocessed_data['kspace_density_compensation'], preprocessed_data['cse'].coil_sens
    return_data = dict()
    return_data[ "kspace_data_z" ] = kspace_data_z
    return_data[ "kspace_data_z_compensated" ] = kspace_data_z * kspace_density_compensation[:,:,None,None,:,:]
    return_data["kspace_density_compensation"] = kspace_density_compensation[:,:,None,None,:,:]
    print(kspace_traj.shape) # torch.Size([34, 5, 2, 15, 640])
    return_data[ "kspace_traj" ] = (kspace_traj[:,:,0]+1j*kspace_traj[:,:,1])[:,:,None,None,:,:] #.expand(kspace_data_z.shape)
    return_data[ "cse" ] = cse

    return return_data


def check_top_k_channel(d, k = 5):
    # d = zarr.open(path, mode='r')
    t, ph, ch, kz, sp, lens = d.shape
    if ch < k:
        return d 
    else:
        center_len  = lens//2
        center_z = kz//2
        lowk_energy = [torch.sum(torch.abs(d[0,0,ch,center_z-5:center_z+5, :, center_len-20:center_len+20])) for ch in range(ch)]
        sorted_energy, sorted_idx = torch.sort(torch.tensor(lowk_energy), descending=True)
        return sorted_idx[:k].tolist()

class Splitted_And_Packed_Dataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs) -> None:
        self.data_dicts = kwargs
    
    def __getitem__(self, idx):
        d = dict()
        for k,v in self.data_dicts.items():
            d[k] = v[idx]
        return d
    
    def __len__(self):
        k = list(self.data_dicts.keys())[0]
        return len(self.data_dicts[k])


class ONC_DCE_DeCoLearn(LightningDataModule):
    def __init__(
        self,
        # data_dir: os.PathLike = '/data-local/anlab/Chunxu/DL_MOTIF/recon_results/CCIR_01168_ONC-DCE',
        data_dir: os.PathLike = '/data-local/anlab/Chunxu/RawData_MR/CCIR_01168_ONC-DCE/',
        train_scope = slice(0,10),
        val_scope = slice(10,11),
        load_scope = slice(0,-1),
        patch_size=(20, 320, 320),
        num_samples_per_subject= 16,
        train_batch_size: int = 4,
        eval_batch_size: int = 1,
        num_workers: int = 0,
        # cache_dir: os.PathLike = '/bmr207/nmrgrp/nmr201/.cache',
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
        self.cache_dir = self.data_dir / '.cache_ONC_DCE_DeCoLearnKXKYZ_5ph'
        self.num_workers = num_workers
        self.contrast, self.phase = 34, 5
        self.top_k = 5
        self.keys = ["kspace_data_z", "kspace_data_z_compensated", "kspace_traj", "kspace_density_compensation"]
        self.val_keys = ["kspace_data_z", "kspace_data_z_compensated", "kspace_traj", "cse"]

        self.dat_file_path_list = [
            'ONC-DCE-001/meas_MID00144_FID02406_CAPTURE_FA15_Dyn.dat',
            'ONC-DCE-003/meas_MID00781_FID11107_CAPTURE_FA15_Dyn.dat',
            'ONC-DCE-004/meas_MID00165_FID04589_Abd_CAPTURE_FA15_Dyn.dat',
            'ONC-DCE-006/meas_MID00221_FID07916_Abd_CAPTURE_FA13_Dyn.dat',
            'ONC-DCE-007/meas_MID00106_FID17478_CAPTURE_FA15_Dyn.dat',
            'ONC-DCE-008/meas_MID00111_FID14538_CAPTURE_FA15_Dyn.dat',
            'ONC-DCE-009/meas_MID00319_FID19874_CAPTURE_FA15_Dyn.dat',
            'ONC-DCE-010/meas_MID00091_FID19991_CAPTURE_FA13_Dyn.dat',
            'ONC-DCE-011/meas_MID00062_FID07015_CAPTURE_FA15_Dyn.dat',
            'ONC-DCE-012/meas_MID00124_FID07996_CAPTURE_FA14_Dyn.dat',
            'ONC-DCE-005/meas_MID01282_FID10023_CAPTURE_FA15_Dyn.dat',
            'ONC-DCE-013/meas_MID00213_FID10842_CAPTURE_FA15_Dyn.dat',
            'ONC-DCE-014/meas_MID00099_FID12331_CAPTURE_FA14_5_Dyn.dat'
        ]

    def prepare_data(self):
        # single thread, download can be done here
        if not iou.check_mk_dirs(self.cache_dir):
            train_group = zarr.group(store = zarr.DirectoryStore(self.cache_dir/"train.zarr"))
            val_group = zarr.group(store = zarr.DirectoryStore(self.cache_dir/"val.zarr"))
            for idx,p in enumerate(self.dat_file_path_list[self.train_scope]):
                dat_file_to_recon = Path(self.data_dir)/p
                raw_data = recon_one_scan(dat_file_to_recon)
                topk_ch = check_top_k_channel(raw_data["kspace_data_z"], k = self.top_k)
                print(topk_ch)
                for k in self.keys:
                    phase = raw_data[k] if raw_data[k].shape[2]==1 else raw_data[k][:,:,topk_ch,:,:,:]
                    if phase.shape[2]==1:
                        d = repeat(phase, 't ph () () sp len -> (t ch) ph () sp len', ch = self.top_k)
                    else:
                        d = rearrange(phase, 't ph ch d sp len -> (t ch) ph d sp len')
                    train_group.require_dataset(k, data=d.numpy(),shape = d.shape) if idx == 0 else train_group[k].append(d.numpy())
            for idx,p in enumerate(self.dat_file_path_list[self.val_scope]):
                dat_file_to_recon = Path(self.data_dir)/p
                raw_data = recon_one_scan(dat_file_to_recon)
                for k in self.val_keys:
                    if k == "cse":
                        d = rearrange(raw_data[k], 'ch d h w -> () ch d h w') 
                    else:
                        d = rearrange(raw_data[k], 't ph ch d sp len  -> () t ch ph d sp len')
                    val_group.require_dataset(k, data=d.numpy(),shape = d.shape) if idx == 0 else val_group[k].append(d.numpy())


    def setup(self, init=False, stage: str = 'train'):
        if stage == 'train' or stage == 'continue':
            data = [zarr.open(self.cache_dir/"train.zarr", mode='r')[k][self.load_scope] for k in self.keys]
            sampler = RandSpatialCropSamplesd(keys=['kspace_data_compensated', 'kspace_data'], roi_size=[
                                                self.patch_size[0], -1, -1], num_samples=self.num_samples_per_subject,random_size=False,)
            train_transforms = Compose([
                ToTensord(device=torch.device('cpu'), keys=[
                   "kspace_data_compensated", "kspace_data", "kspace_traj" 
                ]),
                Lambdad(keys=["kspace_data_compensated", "kspace_data"], func = lambda x: rearrange(x, 'ph z sp len -> ph z (sp len)')),
                Lambdad(keys=["kspace_traj"], func = lambda x: torch.view_as_real(x).to(torch.float32)), 
                Lambdad(keys=["kspace_traj"], func = lambda x: rearrange(x, 'ph () sp len c -> ph c (sp len)')),
                Lambdad(keys=["kspace_density_compensation"], func = lambda x: rearrange(x, 'ph () sp len -> ph () (sp len)')),
            ])
            self.train_dataset = PatchDataset(data = Splitted_And_Packed_Dataset(**dict(zip(["kspace_data", "kspace_data_compensated", "kspace_traj", "kspace_density_compensation"], data))), 
                                                  patch_func=sampler, samples_per_image=self.num_samples_per_subject, 
                                                  transform=train_transforms)
        val_data = dict(
                kspace_data_compensated = zarr.open(self.cache_dir/"val.zarr" , mode='r')["kspace_data_z_compensated"],
                kspace_data = zarr.open(self.cache_dir/"val.zarr" , mode='r')["kspace_data_z"],
                kspace_traj = zarr.open(self.cache_dir/"val.zarr" , mode='r')["kspace_traj"],
                cse = zarr.open(self.cache_dir/"val.zarr" , mode='r')["cse"],
                # scan_id = zarr.open(self.cache_dir/"val.zarr" , mode='r')["scan_id"],
        )
        
        ########## WARNING Load all the 34 contrast will blow up the memory!!! ##########
        eval_transforms = Compose([
                # EnsureChannelFirstd(keys=[
                #     ''], channel_dim="no_channel"),
                Lambdad(keys = ['kspace_data_compensated','kspace_data','kspace_traj', 'cse'],func = lambda x: np.array(x[0:2])),
                ToTensord(device=torch.device('cpu'), keys=['kspace_data_compensated','kspace_data','kspace_traj', 'cse']),
                Lambdad(keys=["kspace_data_compensated", "kspace_data"], func = lambda x: rearrange(x, 't ch ph z sp len -> t ch ph z (sp len)')),
                Lambdad(keys=["kspace_traj"], func = lambda x: torch.view_as_real(x).to(torch.float32)), 
                Lambdad(keys=["kspace_traj"], func = lambda x: rearrange(x, 't ch ph () sp len c -> t ch ph c (sp len)')),
                # Lambdad(keys=["cse"], func = lambda x: rearrange(x, 't ch ph () sp len c -> t ch ph c (sp len)')),
                # Lambdad(keys=['kspace_data_compensated','kspace_data','kspace_traj', 'cse'], func=lambda x: x.unsqueeze(0)),
            ])
        self.val_dataset = Dataset(Splitted_And_Packed_Dataset(**val_data), transform=eval_transforms)
      
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle = True, num_workers=20,pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, collate_fn=self.transfer_batch_to_device,)

if __name__ == "__main__":
    # dataset = LibriSpeech()
    # dataset.prepare_data()

    data = ONC_DCE_DeCoLearn(train_scope=slice(0,10), val_scope=slice(10,11), load_scope=slice(0,1))
    data.prepare_data()
    data.setup()
    # for i in data.train_dataloader():
    #     print(i.keys())
# %%
