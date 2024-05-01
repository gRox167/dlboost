import os

import numpy as np
import torch
from icecream import ic
from lightning.pytorch.callbacks import BasePredictionWriter
from nibabel import load
from torch import multiprocessing as mp


def async_save_xarray_dataset(
    ds, save_path, client, mode=None, group=None, region=None
):
    write_task = ds.to_zarr(
        save_path, compute=False, mode=mode, group=group, region=region
    )
    future = client.compute(write_task)
    return future


def read_analyze_format(path):
    img = load(path)
    data_array = img.get_fdata()
    return img.header, data_array


def check_mk_dirs(paths):
    if isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)
    return paths


def from_label_to_onehot(labels, num_classes):
    one_hot = torch.zeros(
        labels.size(0), num_classes, labels.size(2), labels.size(3), labels.size(4)
    ).to(labels.device)
    target = one_hot.scatter_(1, labels.to(torch.int64), 1)
    return target


def abs_helper(x, axis=1, is_normalization=True):
    x = torch.sqrt(torch.sum(x**2, dim=axis, keepdim=True))

    if is_normalization:
        for i in range(x.shape[0]):
            x[i] = (x[i] - torch.min(x[i])) / (
                torch.max(x[i]) - torch.min(x[i]) + 1e-16
            )

    x = x.to(torch.float32)

    return x


def check_and_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def dict2pformat(x: dict):
    ret = ""
    for k in x:
        ret += " %s: [%.4f]" % (k, x[k])
    return ret


def dict2md_table(ipt: dict):
    ret = str()
    for section in ipt.keys():
        ret += "## " + section + "\n"
        ret += "|  Key  |  Value |\n|:----:|:---:|\n"

        for i in ipt[section].keys():
            ret += "|" + i + "|" + str(ipt[section][i]) + "|\n"

        ret += "\n\n"

    return ret


def write_test(log_dict, img_dict, save_path, is_save_mat=False, is_save_tiff=True):
    if log_dict:
        # Write Log_dict Information
        cvs_data = np.array(list(log_dict.values()))
        cvs_data = np.transpose(cvs_data, [1, 0])

        cvs_data_mean = cvs_data.mean(0)
        cvs_data_mean.shape = [1, -1]

        num_index = cvs_data.shape[0]
        cvs_index = np.arange(num_index) + 1
        cvs_index.shape = [-1, 1]

        cvs_data_with_index = np.concatenate([cvs_index, cvs_data], 1)

        cvs_header = ""
        for k in log_dict:
            cvs_header = cvs_header + k + ","

        np.savetxt(
            save_path + "metrics.csv",
            cvs_data_with_index,
            delimiter=",",
            fmt="%.5f",
            header="index," + cvs_header,
        )
        np.savetxt(
            save_path + "metrics_mean.csv",
            cvs_data_mean,
            delimiter=",",
            fmt="%.5f",
            header=cvs_header,
        )

        print(cvs_data_mean)

    if is_save_tiff:
        # Write recon of Img_Dict Information
        for key_ in [
            "fixed_y_tran",
            "fixed_y_tran_recon",
            "fixed_x",
            "moved_y_tran",
            "moved_y_tran_recon",
            "moved_x",
            "wrapped_f2m",
            "wrapped_m2f",
        ]:
            if key_ in img_dict:
                print(key_, img_dict[key_].shape)
                to_tiff(img_dict[key_], save_path + key_ + ".tiff", is_normalized=False)


def multi_processing_save_data(data, save_func):
    queue = mp.Queue()

    def task(queue):
        while True:
            data_remote = queue.get()
            if data_remote is None:
                break
            save_func(data_remote)

    writer_process = mp.Process(target=task, args=(queue,))
    writer_process.start()
    queue.put(data)
    queue.put(None)
    return writer_process, queue


if __name__ == "__main__":
    # test for heatmap
    # mask = tio.ScalarImage(
    #     path='careIIChallenge/preprocessed/mask_private/125.nii.gz')
    # mask_to_heatmap(mask.data)
    # mask_to_polygon((mask.data[0, :, :, 172]))
    labels = torch.ones((2, 1, 5, 5, 5))
    onehot = from_label_to_onehot(labels, 5)
    print(onehot.shape)
