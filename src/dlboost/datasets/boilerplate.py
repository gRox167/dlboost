import torch
from mrboost import io_utils as iou
from mrboost import reconstruction as recon
from mrboost import computation as comp


def recon_one_scan(dat_file_to_recon, phase_num=5, time_per_contrast=10):
    reconstructor = recon.CAPTURE_VarW_NQM_DCE_PostInj(
        dat_file_location=dat_file_to_recon, phase_num=phase_num, which_slice=-1, which_contra=-1, which_phase=-1, time_per_contrast=time_per_contrast, device=torch.device('cuda:0'))
    raw_data = reconstructor.get_raw_data(reconstructor.dat_file_location)
    reconstructor.args_init()

    preprocessed_data = reconstructor.data_preprocess(raw_data)
    kspace_data_centralized, kspace_data_mask, kspace_data_z, kspace_traj, kspace_density_compensation, cse =\
        preprocessed_data['kspace_data_centralized'], preprocessed_data['kspace_data_mask'], preprocessed_data['kspace_data_z'], \
        preprocessed_data['kspace_traj'], preprocessed_data['kspace_density_compensation'], preprocessed_data['cse'].coil_sens
    return_data = dict()
    return_data["kspace_data_z"] = comp.normalization_root_of_sum_of_square(
        kspace_data_z)
    return_data["kspace_data_z_compensated"] = comp.normalization_root_of_sum_of_square(
        kspace_data_z * kspace_density_compensation[:, :, None, None, :, :]*1000)
    return_data["kspace_density_compensation"] = kspace_density_compensation[:,
                                                                             :, None, None, :, :]
    print(kspace_traj.shape)  # torch.Size([34, 5, 2, 15, 640])
    return_data["kspace_traj"] = (kspace_traj[:, :, 0]+1j*kspace_traj[:, :, 1])[
        :, :, None, None, :, :]  # .expand(kspace_data_z.shape)
    return_data["cse"] = cse

    return return_data


def check_top_k_channel(d, k=5):
    # d = zarr.open(path, mode='r')
    t, ph, ch, kz, sp, lens = d.shape
    if ch < k:
        return d
    else:
        center_len = lens//2
        center_z = kz//2
        lowk_energy = [torch.sum(torch.abs(
            d[0, 0, ch, center_z-5:center_z+5, :, center_len-20:center_len+20])) for ch in range(ch)]
        sorted_energy, sorted_idx = torch.sort(
            torch.tensor(lowk_energy), descending=True)
        return sorted_idx[:k].tolist()


class Splitted_And_Packed_Dataset(torch.utils.data.Dataset):
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
