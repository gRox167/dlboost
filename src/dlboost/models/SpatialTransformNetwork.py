import torch
import torch.nn as nn
import torch.nn.functional as F

from dlboost.utils.tensor_utils import (
    GridSample3dBackward,
    GridSample3dForward,
)


class SpatialTransformNetwork(nn.Module):
    def __init__(self, size, mode="bilinear", dims=3):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer("grid", grid)
        self.grid_sampling = GridSample3dForward if dims == 3 else None
        self.grid_sampling_backward = GridSample3dBackward if dims == 3 else None

    def forward(self, src, flow, return_phi=False):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        with torch.amp.autocast("cuda", enabled=False):
            new_locs = new_locs.float()
            src = src.float()
            if return_phi:
                return self.grid_sampling.apply(src, new_locs, True), new_locs
            else:
                return self.grid_sampling.apply(src, new_locs, True)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, "nsteps should be >= 0, found: %d" % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2**self.nsteps)
        self.transformer = SpatialTransformNetwork(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class CompositionTransform(nn.Module):
    def __init__(self, size, mode="bilinear", dims=3):
        super().__init__()
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

    def forward(self, flow_1, flow_2, sample_grid, range_flow):
        size_tensor = sample_grid.size()
        grid = sample_grid + (flow_2.permute(0, 2, 3, 4, 1) * range_flow)
        grid[0, :, :, :, 0] = (
            (grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2))
            / (size_tensor[3] - 1)
            * 2
        )
        grid[0, :, :, :, 1] = (
            (grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2))
            / (size_tensor[2] - 1)
            * 2
        )
        grid[0, :, :, :, 2] = (
            (grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2))
            / (size_tensor[1] - 1)
            * 2
        )
        compos_flow = (
            F.grid_sample(flow_1, grid, mode="bilinear", align_corners=True) + flow_2
        )
        return compos_flow


def warp(
    moving_image,
    displacement_field,
    mass_preserving=False,
    oversample_ratio=(2, 2, 2),
    direction_dim=-1,
):
    """
    Warps a 3D image using a displacement field.
    Args:
        moving_image (torch.Tensor): The image to be warped, shape (B, C, D, H, W).
        displacement_field (torch.Tensor): The displacement field, shape (B, D, H, W, 3). Last dimension contains the displacement in x, y, z order. Range should be [-1, 1] for each dimension.
    Returns:
        torch.Tensor: The warped image, shape (B, C, D, H, W).
    """
    assert displacement_field.shape[direction_dim] == 3, (
        f"Expected displacement_field to have 3 channels at dimension {direction_dim}, but got {displacement_field.shape[direction_dim]}"
    )
    with torch.amp.autocast("cuda", enabled=False):
        if oversample_ratio is not None:
            moving_image_os = F.interpolate(
                moving_image,
                scale_factor=oversample_ratio,
                mode="trilinear",
                align_corners=True,
            )
            displacement_field_os = F.interpolate(
                displacement_field.permute(
                    0, 4, 1, 2, 3
                ),  # move last dim to second position
                scale_factor=oversample_ratio,
                mode="trilinear",
                align_corners=True,
            ).permute(0, 2, 3, 4, 1)  # move back to original order
        else:
            moving_image_os = moving_image
            displacement_field_os = displacement_field
        # new locations
        grid = F.affine_grid(
            torch.eye(3, 4, device=moving_image_os.device)[None],
            (1, 1) + moving_image_os.shape[2:],
            align_corners=True,
        )
        warped_coordinates = grid + displacement_field_os
        moved_image_os = F.grid_sample(
            moving_image_os, warped_coordinates, "bilinear", align_corners=True
        )
        if oversample_ratio is not None:
            # interpolate back to original resolution
            moved_image = F.interpolate(
                moved_image_os,
                scale_factor=tuple(1 / s for s in oversample_ratio),
                mode="trilinear",
                align_corners=True,
            )
        else:
            moved_image = moved_image_os
        return moved_image
    # if mass_preserving:
    #     J = J = jacobian(warped_coordinates, normalize=True).permute(0, 2, 3, 4, 1, 5)[
    #         :, 1:-1, 1:-1, 1:-1, :
    #     ]
    #     Jdet = torch.linalg.det(J)
    #     # if mass preserving, we need to scale the image by the inverse of determinant of the Jacobian
    #     moved_image_os[:, :, 1:-1, 1:-1, 1:-1] /= Jdet.unsqueeze(1) + 1e-6
    # return GridSample3dForward.apply(moving_image, warped_coordinates, True)

    # warped_coordinates = einx.rearrange(
    #     "b v ... -> b ... v", grid + displacement_field
    # ).flip(-1)  # rearrange to match grid_sample input
    # flip last dimention from zyx to xyz order
    # if return_warped_coordinates:
    #     return GridSample3dForward.apply(
    #         moving_image, warped_coordinates, True
    #     ), warped_coordinates
    # else:
