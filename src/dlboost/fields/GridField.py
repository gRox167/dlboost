from typing import Sequence

import einx
import torch
from plum import dispatch
from torch import nn


class Grid(nn.Module):
    def __init__(self, size, dtype=torch.complex64):
        super().__init__()
        self.grid = nn.Parameter(
            0.001 * torch.ones(size, dtype=dtype),
            requires_grad=True,
        )

    def reconstruct(self, total_step_number):
        return self.grid

    # @dispatch
    # def forward(self, index: int):
    #     return self.grid[index]  # Remove the batch dimension

    # @dispatch
    def forward(self, slices: Sequence[slice]):
        f_list = []
        for s in slices:
            t, x, y, z = s
            ft = self.grid[t]
            f_list.append(ft)
        return torch.stack(f_list, dim=0)


class StaticDynamicGrid(nn.Module):
    def __init__(self, size):
        super().__init__()
        t, x, y, z = size
        self.static_grid = Grid((x, y, z))
        self.dynamic_grid = Grid((t, x, y, z))

    def forward(self, slices: Sequence[slice]):
        # Ensure coordinates are in the correct shape and normalized
        # coordinates = coordinates.unsqueeze(0)  # Add channel dimensions
        # coordinates = 2.0 * coordinates - 1.0  # Normalize to [-1, 1]

        # Sample the grid with the given coordinates

        # sampled_grid = F.grid_sample(
        #     self.grid.unsqueeze(0), coordinates, mode="trilinear", align_corners=True
        # )
        output = []
        for s in slices:
            t, x, y, z = s
            output.append(
                self.static_grid.grid[x, y, z] + self.dynamic_grid.grid[t, x, y, z]
            )
        return torch.stack(output, dim=0)


class ResidualGrid(nn.Module):
    def __init__(self, size):
        super().__init__()
        t, x, y, z = size
        self.t = t
        self.x0 = nn.Parameter(
            0.001 * torch.rand((x, y, z, 2), dtype=torch.float32),
            requires_grad=True,
        )
        self.delta = nn.Parameter(
            0.001 * torch.rand((t, x, y, z, 2), dtype=torch.float32),
            requires_grad=True,
        )

    @dispatch
    def get_image(self, indices: Sequence[int]):
        images_forward = []
        images_backward = []
        for i in indices:
            f, b = self.get_image(i)
            images_forward.append(f)
            images_backward.append(b)
        return torch.stack(images_forward, dim=0), torch.stack(images_backward, dim=0)

    @dispatch
    def get_image(self, index: int):
        # if index == 0:
        # return self.x0
        # diff_last_to_first = -self.delta.sum(dim=0)
        # full_delta = torch.cat([self.delta, diff_last_to_first.unsqueeze(0)], dim=0)
        forward_cumulation = self.delta[0:index].sum(dim=0)
        backward_cumulation = self.delta[index:].sum(dim=0)
        return torch.view_as_complex(
            self.x0 + forward_cumulation
        ), torch.view_as_complex(self.x0 - backward_cumulation)

    def forward(self, slices, double_direction=True):
        cum_sum = torch.cumsum(self.delta, dim=0)
        rev_cum_sum = self.delta - cum_sum + cum_sum[-1]

        images_forward = einx.rearrange(
            "x y z cmp, t x y z cmp -> (1 + t) x y z cmp",
            self.x0,
            self.x0 + cum_sum[:-1],
        )
        f_list = []
        if double_direction:
            images_backward = self.x0 - rev_cum_sum
            b_list = []
        for s in slices:
            t, x, y, z = s
            f_list.append(images_forward[t])
            if double_direction:
                b_list.append(images_backward[t])
        if double_direction:
            return torch.view_as_complex(
                torch.stack(f_list, dim=0)
            ), torch.view_as_complex(torch.stack(b_list, dim=0))

        else:
            return torch.view_as_complex(torch.stack(f_list, dim=0))


class FourierGrid(nn.Module):
    def __init__(self, size, fourier_components=10):
        super().__init__()
        t, x, y, z = size
        self.t = t
        self.fourier_components = fourier_components
        self.dynamic_fourier_grid = nn.Parameter(
            0.001 * torch.rand((fourier_components, x, y, z), dtype=torch.complex64),
            requires_grad=True,
        )

    @dispatch
    def get_image(self, indices: Sequence[int]):
        # images_forward = []
        dynamic_grid = torch.fft.fft(
            self.dynamic_fourier_grid, n=self.t, dim=0, norm="ortho"
        )[indices]
        return dynamic_grid

    @dispatch
    def get_image(self, index: int):
        dynamic_grid = torch.fft.ifft(
            self.dynamic_fourier_grid, n=self.t, dim=0, norm="ortho"
        )[index]
        return dynamic_grid

    def reconstruct(self, total_step_number):
        f = self.dynamic_fourier_grid
        t_pad = (total_step_number - self.fourier_components) // 2
        z = torch.zeros(t_pad, *f.shape[1:], dtype=f.dtype, device=f.device)
        f_double_side_padded = torch.cat([z, f, z], dim=0)
        dynamic_grid = torch.fft.ifft(
            torch.fft.ifftshift(f_double_side_padded, dim=0),
            dim=0,
            norm="ortho",
        )
        return dynamic_grid

    def forward(self, slices, total_step_number):
        f_list = []
        dynamic_grid = torch.fft.ifft(
            self.dynamic_fourier_grid, n=total_step_number, dim=0, norm="ortho"
        )
        for s in slices:
            t, x, y, z = s
            # t_step = 1 if t.step is None else t.step
            f = dynamic_grid[t]
            f_list.append(f)
        return torch.stack(f_list, dim=0)


class ContrastODE(nn.Module):
    def __init__(self, size, fourier_components=16):
        super().__init__()
        t, x, y, z = size
        self.t = t
        self.fourier_components = fourier_components
        # self.x0 = nn.Parameter(
        #     0.001 * torch.rand((x, y, z), dtype=torch.complex64),
        #     requires_grad=True,
        # )
        if fourier_components:
            self.x0 = Grid((1, x, y, z), dtype=torch.complex64)
            self.c_latent = FourierGrid(
                (t - 1, x, y, z),
                fourier_components=fourier_components,
            )
        else:
            self.c_latent = Grid((t, x, y, z), dtype=torch.complex64)
        # contrast enhancement rate

    @dispatch
    def get_image(self, indices: Sequence[int]):
        images_forward = []
        images_backward = []
        for i in indices:
            f, b = self.get_image(i)
            images_forward.append(f)
            images_backward.append(b)
        return torch.stack(images_forward, dim=0), torch.stack(images_backward, dim=0)

    @dispatch
    def get_image(self, index: int):
        c = self.c_latent.reconstruct(self.t)
        if self.fourier_components:
            c = torch.cat((self.x0.reconstruct(self.t), c), dim=0)
        f = self.euler_solver(None, c, index)
        return f

    @dispatch
    def euler_solver(self, x0, c):
        return torch.cumsum(c, dim=0)
        # x = x0
        # x = x0 + torch.cumsum(c, dim=0)
        # return x

    @dispatch
    def euler_solver(self, x0, c, t):
        return torch.sum(c[: t + 1], dim=0)
        # return x0 + torch.sum(c[:t], dim=0)

    def forward(self, slices, total_step_number):
        c = self.c_latent.reconstruct(total_step_number)
        if self.fourier_components:
            c = torch.cat((self.x0.reconstruct(self.t), c), dim=0)
        # f = self.euler_solver(self.x0, c)
        f = self.euler_solver(None, c)
        f_list = []
        for s in slices:
            t, x, y, z = s
            ft = f[t]
            f_list.append(ft)
        return torch.stack(f_list, dim=0)
