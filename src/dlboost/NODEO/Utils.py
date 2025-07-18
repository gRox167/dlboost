from dataclasses import dataclass

import einx
import nibabel as nib
import numpy as np

# from dlboost.NODEO.Loss import
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode="bilinear"):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer("grid", grid)

    def forward(self, src, flow, return_phi=False):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position channel have the order of (z, y, x)
        # because grid_sample expects the last dimension to specify the channel
        # the grid_sample function expects ordering (x, y, z)
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        if return_phi:
            return F.grid_sample(
                src,
                new_locs,
                align_corners=True,
                mode=self.mode,
                # src, new_locs, align_corners=False, mode=self.mode
            ), new_locs
        else:
            return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
            # return F.grid_sample(src, new_locs, align_corners=False, mode=self.mode)


def resize_deformation_field(field, factor, ndims=3):
    """
    Resize a deformation field.
    """
    if ndims == 2:
        mode = "bilinear"
    elif ndims == 3:
        mode = "trilinear"
    else:
        raise ValueError("Only 2D and 3D supported")
    _field = F.interpolate(field, scale_factor=factor, mode=mode, align_corners=False)
    return einx.multiply(
        "b [c] ...", _field, torch.tensor(factor, device=_field.device)
    )


def load_nii(path):
    X = nib.load(path)
    X = X.get_fdata()
    return X


def save_nii(img, savename):
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(img, affine, header=None)
    nib.save(new_img, savename)


def generate_grid3D_tensor(shape):
    x_grid = torch.linspace(-1.0, 1.0, shape[0])
    y_grid = torch.linspace(-1.0, 1.0, shape[1])
    z_grid = torch.linspace(-1.0, 1.0, shape[2])
    x_grid, y_grid, z_grid = torch.meshgrid(x_grid, y_grid, z_grid)

    # Note that default the dimension in the grid is reversed:
    # z, y, x
    grid = torch.stack([z_grid, y_grid, x_grid], dim=0)
    return grid


def dice(array1, array2, labels):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.
    """
    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem


@dataclass
class Config:
    # registration
    smoothing_kernel = "AK"
    smoothing_win: int = 15
    smoothing_pass: int = 1
    ds: int = 2
    bs: int = 16
    # ode
    time_steps: int = 2
    STEP_SIZE: float = 0.001
    optimizer: str = "Euler"
    lr: float = 0.005
    # loss
    NCC_win: int = 21
    lambda_v: float = 0.00005
    lambda_J: float = 2.5
    lambda_df: float = 0.05
    # training
    epoches: int = 300
