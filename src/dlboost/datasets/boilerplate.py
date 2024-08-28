from os import PathLike
from pathlib import Path
from typing import Literal

import monai
import torch

# from monai.data.image_reader import ImageReader
import zarr
from einops import rearrange  # , reduce, repeat
from lightning.pytorch.callbacks import BasePredictionWriter
from optree import tree_structure, tree_transpose
from torch.utils.data import Dataset

from mrboost import computation as comp
from mrboost import reconstruction as recon


def recon_one_scan(dat_file_to_recon, phase_num=5, time_per_contrast=10):
    reconstructor = recon.CAPTURE_VarW_NQM_DCE_PostInj(
        dat_file_location=dat_file_to_recon,
        phase_num=phase_num,
        which_slice=-1,
        which_contra=-1,
        which_phase=-1,
        time_per_contrast=time_per_contrast,
        device=torch.device("cuda:1"),
    )
    raw_data = reconstructor.get_raw_data(reconstructor.dat_file_location)
    reconstructor.args_init()

    preprocessed_data = reconstructor.data_preprocess(raw_data)
    _, _, kspace_data_z, kspace_traj, kspace_density_compensation, cse = (
        preprocessed_data["kspace_data_centralized"],
        preprocessed_data["kspace_data_mask"],
        preprocessed_data["kspace_data_z"],
        preprocessed_data["kspace_traj"],
        preprocessed_data["kspace_density_compensation"],
        preprocessed_data["cse"].coil_sens,
    )
    return_data = dict()
    return_data["kspace_data_z"] = comp.normalization_root_of_sum_of_square(
        kspace_data_z
    )
    return_data["kspace_data_z_compensated"] = (
        comp.normalization_root_of_sum_of_square(
            kspace_data_z
            * kspace_density_compensation[:, :, None, None, :, :]
            * 1000
        )
    )
    return_data["kspace_density_compensation"] = kspace_density_compensation[
        :, :, None, None, :, :
    ]
    # print(kspace_traj.shape)  # torch.Size([34, 5, 2, 15, 640])
    # return_data["kspace_traj"] = (kspace_traj[:, :, 0]+1j*kspace_traj[:, :, 1])[
    #     :, :, None, None, :, :]
    return_data["kspace_traj"] = kspace_traj
    return_data["cse"] = cse
    # print(return_data["cse"].shape) # [15, 80, 320, 320]
    return return_data


def recon_one_scan_P2PDeCo(
    dat_file_to_recon, phase_num=5, time_per_contrast=10
):
    reconstructor = recon.CAPTURE_VarW_NQM_DCE_PostInj(
        dat_file_location=dat_file_to_recon,
        phase_num=phase_num,
        which_slice=-1,
        which_contra=-1,
        which_phase=-1,
        time_per_contrast=time_per_contrast,
        device=torch.device("cuda:1"),
    )
    raw_data = reconstructor.get_raw_data(reconstructor.dat_file_location)
    reconstructor.args_init()

    preprocessed_data = reconstructor.data_preprocess(raw_data)
    _, _, kspace_data_z, kspace_traj, kspace_density_compensation, cse = (
        preprocessed_data["kspace_data_centralized"],
        preprocessed_data["kspace_data_mask"],
        preprocessed_data["kspace_data_z"],
        preprocessed_data["kspace_traj"],
        preprocessed_data["kspace_density_compensation"],
        preprocessed_data["cse"].coil_sens,
    )
    return_data = dict()
    return_data["kspace_data_z"] = comp.normalization_root_of_sum_of_square(
        kspace_data_z
    )
    return_data["kspace_data_z_compensated"] = (
        comp.normalization_root_of_sum_of_square(
            kspace_data_z
            * kspace_density_compensation[:, :, None, None, :, :]
            * 1000
        )
    )
    return_data["kspace_density_compensation"] = kspace_density_compensation[
        :, :, None, None, :, :
    ]
    # print(kspace_traj.shape)  # torch.Size([34, 5, 2, 15, 640])
    # return_data["kspace_traj"] = (kspace_traj[:, :, 0]+1j*kspace_traj[:, :, 1])[
    #     :, :, None, None, :, :]
    return_data["kspace_traj"] = kspace_traj
    return_data["cse"] = cse
    # print(return_data["cse"].shape) # [15, 80, 320, 320]
    return return_data


def recon_one_scan_BlackBone(
    dat_file_location, motion_curve_location, device=torch.device("cuda:1")
):
    r = recon.BlackBone_LowResFrames(
        dat_file_location, motion_curve_location, device=device
    )
    data_raw = r.get_raw_data(r.dat_file_location)
    r.args_init()
    data_dict = r.data_preprocess(data_raw)
    kspace_data_z, kspace_traj, kspace_density_compensation, cse = (
        data_dict["kspace_data_z"],
        data_dict["kspace_traj"],
        data_dict["kspace_density_compensation"],
        data_dict["cse"],
    )
    return_data = dict()
    return_data["kspace_data_z"] = kspace_data_z
    return_data["kspace_density_compensation"] = kspace_density_compensation[
        :, :, None, None, :, :
    ]
    return_data["kspace_traj"] = kspace_traj
    return_data["cse"] = cse


def collate_fn(batch):
    # transpose dict structure out of list
    batch_transposed = tree_transpose(
        tree_structure([0 for _ in batch]), tree_structure(batch[0]), batch
    )
    return {k: torch.stack(v) for k, v in batch_transposed.items()}


def check_top_k_channel(d, k=5):
    # d = zarr.open(path, mode='r')
    t, ph, ch, kz, sp, lens = d.shape
    if ch < k:
        return d
    else:
        center_len = lens // 2
        center_z = kz // 2
        lowk_energy = [
            torch.sum(
                torch.abs(
                    d[
                        0,
                        0,
                        ch,
                        center_z - 5 : center_z + 5,
                        :,
                        center_len - 20 : center_len + 20,
                    ]
                )
            )
            for ch in range(ch)
        ]
        sorted_energy, sorted_idx = torch.sort(
            torch.tensor(lowk_energy), descending=True
        )
        return sorted_idx[:k].tolist()


class Splitted_And_Packed_Dataset(monai.data.Dataset):
    def __init__(self, **kwargs) -> None:
        self.data_dicts = kwargs

    def __getitem__(self, idx):
        d = dict()
        for k, v in self.data_dicts.items():
            d[k] = v[idx]
        return d

    def __len__(self):
        k = list(self.data_dicts.keys())[0]
        return len(self.data_dicts[k])


class DCE_P2PCSE_KXKY_Dataset(Dataset):
    def __init__(self, data, transform=None, mode="train") -> None:
        super().__init__()
        self.keys = [
            "kspace_data_z",
            "kspace_data_z_compensated",
            "kspace_traj",
            "kspace_density_compensation",
        ]
        self.data = data
        self.mode = mode

    def __len__(self) -> int:
        if self.mode == "train":
            return self.data["kspace_data_z"].shape[2]
        else:
            return len(self.data)

    def __getitem__(self, index: int):
        if self.mode == "train":
            kspace_data_z = self.data["kspace_data_z"][:, :, index, ...]
            kspace_data_z_compensated = self.data["kspace_data_z_compensated"][
                :, :, index, ...
            ]
            kspace_traj = torch.from_numpy(self.data["kspace_traj"][:, :, 0])
            kspace_data_z = rearrange(
                torch.from_numpy(kspace_data_z),
                "ph ch sp len -> ph ch (sp len)",
            )
            kspace_data_z_compensated = rearrange(
                torch.from_numpy(kspace_data_z_compensated),
                "ph ch sp len -> ph ch (sp len)",
            )
            # kspace_data_z_cse = rearrange(
            #     torch.from_numpy(kspace_data_z_compensated[..., 240:400]), 'ph ch sp len -> ph ch (sp len)')

            # print(self.data["kspace_traj"].shape)
            kspace_traj = torch.view_as_real(kspace_traj).to(torch.float32)
            kspace_traj = rearrange(
                kspace_traj, "ph () sp len c -> ph c (sp len)"
            )
            return dict(
                kspace_data_z=kspace_data_z,
                kspace_data_z_compensated=kspace_data_z_compensated,
                # cse = self.data[index]["cse"],
                # kspace_density_compensation = self.data["kspace_density_compensation"][:,:,0],
                kspace_traj=kspace_traj,
            )
        else:
            kspace_data_z = self.data[index]["kspace_data_z"][:]
            kspace_data_z_compensated = self.data[index][
                "kspace_data_z_compensated"
            ][:]
            kspace_traj = torch.from_numpy(
                self.data[index]["kspace_traj"][:, :, 0]
            )  # ph, ch=1, z=1, sp, len
            kspace_data_z = rearrange(
                torch.from_numpy(kspace_data_z),
                "ph ch z sp len -> ph ch z (sp len)",
            )
            kspace_data_z_compensated = rearrange(
                torch.from_numpy(kspace_data_z_compensated),
                "ph ch z sp len -> ph ch z (sp len)",
            )
            # print(self.data["kspace_traj"].shape)
            kspace_traj = torch.view_as_real(kspace_traj).to(torch.float32)
            kspace_traj = rearrange(
                kspace_traj, "ph () sp len c -> ph c (sp len)"
            )
            return dict(
                kspace_data_z=kspace_data_z,
                kspace_data_z_compensated=kspace_data_z_compensated,
                # cse = self.data[index]["cse"],
                # kspace_density_compensation = self.data["kspace_density_compensation"][:,:,0],
                kspace_traj=kspace_traj,
            )

    def __getitems__(self, indices):
        return [self.__getitem__(index) for index in indices]


class DCE_P2PCSE_Writer(BasePredictionWriter):
    def __init__(
        self,
        output_dir: PathLike,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
    ) -> None:
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        predictions,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ) -> None:
        for d in predictions:
            print(d["image_init"].shape)
            print(d["image_recon"].shape)
            pid, contrast = d["id"].split("_")
            zarr.save(
                self.output_dir / "MCNUFFT" / f"{pid}" / f"{contrast}.zarr",
                d["image_init"].abs().numpy(force=True),
            )
            zarr.save(
                self.output_dir / "MOTIF-CORD" / f"{pid}" / f"{contrast}.zarr",
                d["image_recon"].abs().numpy(force=True),
            )
            # to_nifty(torch.swapdims(d["image_init"].abs(),0,-1), self.output_dir/f"{d['id']}_init.nii.gz")
            # breakpoint()
            # to_nifty(torch.swapdims(d["image_recon"].abs(),0,-1), self.output_dir/f"{d['id']}_recon.nii.gz")
            # nib.save(d["image_init"].numpy(force=True), self.output_dir/f"{d['id']}_init.nii.gz")
            # nib.save(d["image_recon"].numpy(force=True), self.output_dir/f"{d['id']}_recon.nii.gz")
