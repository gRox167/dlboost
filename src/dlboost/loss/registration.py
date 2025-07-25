import einx
import torch
import torch.nn.functional as F
from icecream import ic


class MVFLoss(torch.nn.Module):
    def __init__(self, scale, magnitude_coef=None, jacobian_coef=2.5, smooth_coef=0.5):
        super(MVFLoss, self).__init__()
        # self.magnitude_coef = magnitude_coef
        self.scale = scale
        self.jacobian_coef = jacobian_coef
        self.smooth_coef = smooth_coef

    def forward(self, df):
        # Generate 3D grid within range (-1,1) that has the same size as the displacement field df
        # If df has six dimensions, rearrange it so that the displacement part is at dims 2,3,4
        if df.dim() == 6:
            df = einx.rearrange("b ph c d h w -> (b ph) c d h w", df)
        d, h, w = df.shape[-3:]
        grid_d = torch.linspace(-1, 1, steps=d, device=df.device, dtype=df.dtype)
        grid_h = torch.linspace(-1, 1, steps=h, device=df.device, dtype=df.dtype)
        grid_w = torch.linspace(-1, 1, steps=w, device=df.device, dtype=df.dtype)
        meshgrid = torch.meshgrid(grid_d, grid_h, grid_w, indexing="ij")
        grid = torch.stack(meshgrid, dim=0)  # shape: (3, d, h, w)
        grid = grid.unsqueeze(0)  # shape: (1, 3, d, h, w)
        jac_loss = (
            ic(self.jacobian_coef * neg_Jdet_loss(df + grid))
            if self.jacobian_coef is not None
            else 0
        )
        return self.scale * (
            jac_loss
            # ic(self.magnitude_coef * magnitude_loss(df))
            + ic(self.smooth_coef * smooth_l2_loss(df))
        )


def JacboianDet(J):
    if J.size(-1) != 3:
        J = J.permute(0, 2, 3, 4, 1)
    J = J + 1
    J = J / 2.0
    scale_factor = (
        torch.tensor([J.size(1), J.size(2), J.size(3)]).to(J).view(1, 1, 1, 1, 3) * 1.0
    )
    J = J * scale_factor

    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:, :, :, :, 0] * (
        dy[:, :, :, :, 1] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 1]
    )
    Jdet1 = dx[:, :, :, :, 1] * (
        dy[:, :, :, :, 0] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 0]
    )
    Jdet2 = dx[:, :, :, :, 2] * (
        dy[:, :, :, :, 0] * dz[:, :, :, :, 1] - dy[:, :, :, :, 1] * dz[:, :, :, :, 0]
    )

    Jdet = Jdet0 - Jdet1 + Jdet2
    return Jdet


def neg_Jdet_loss(J):
    Jdet = JacboianDet(J)
    neg_Jdet = -1.0 * (Jdet - 0.5)
    selected_neg_Jdet = F.relu(neg_Jdet)
    return torch.mean(selected_neg_Jdet**2)


def smooth_l2_loss(df, dim_ratio=(1, 1, 1)):
    return (
        dim_ratio[0] * ((df[:, :, 1:, :, :] - df[:, :, :-1, :, :]) ** 2).mean()
        + dim_ratio[1] * ((df[:, :, :, 1:, :] - df[:, :, :, :-1, :]) ** 2).mean()
        + dim_ratio[2] * ((df[:, :, :, :, 1:] - df[:, :, :, :, :-1]) ** 2).mean()
    )


def magnitude_loss(all_v):
    all_v_x_2 = all_v[:, :, 0, :, :, :] * all_v[:, :, 0, :, :, :]
    all_v_y_2 = all_v[:, :, 1, :, :, :] * all_v[:, :, 1, :, :, :]
    all_v_z_2 = all_v[:, :, 2, :, :, :] * all_v[:, :, 2, :, :, :]
    all_v_magnitude = torch.mean(all_v_x_2 + all_v_y_2 + all_v_z_2)
    return all_v_magnitude


class NCC(torch.nn.Module):
    """
    NCC with cumulative sum implementation for acceleration. local (over window) normalized cross correlation.
    """

    def __init__(self, win=21, eps=1e-5):
        super(NCC, self).__init__()
        self.eps = eps
        self.win = win
        self.win_raw = win

    def window_sum_cs3D(self, I, win_size):
        half_win = int(win_size / 2)
        pad = [half_win + 1, half_win] * 3

        I_padded = F.pad(I, pad=pad, mode="constant", value=0)  # [x+pad, y+pad, z+pad]

        # Run the cumulative sum across all 3 dimensions
        I_cs_x = torch.cumsum(I_padded, dim=2)
        I_cs_xy = torch.cumsum(I_cs_x, dim=3)
        I_cs_xyz = torch.cumsum(I_cs_xy, dim=4)

        x, y, z = I.shape[2:]

        # Use subtraction to calculate the window sum
        I_win = (
            I_cs_xyz[:, :, win_size:, win_size:, win_size:]
            - I_cs_xyz[:, :, win_size:, win_size:, :z]
            - I_cs_xyz[:, :, win_size:, :y, win_size:]
            - I_cs_xyz[:, :, :x, win_size:, win_size:]
            + I_cs_xyz[:, :, win_size:, :y, :z]
            + I_cs_xyz[:, :, :x, win_size:, :z]
            + I_cs_xyz[:, :, :x, :y, win_size:]
            - I_cs_xyz[:, :, :x, :y, :z]
        )

        return I_win

    def forward(self, I, J):
        # compute CC squares
        I = I.double()
        J = J.double()

        I2 = I * I
        J2 = J * J
        IJ = I * J

        # compute local sums via cumsum trick
        I_sum_cs = self.window_sum_cs3D(I, self.win)
        J_sum_cs = self.window_sum_cs3D(J, self.win)
        I2_sum_cs = self.window_sum_cs3D(I2, self.win)
        J2_sum_cs = self.window_sum_cs3D(J2, self.win)
        IJ_sum_cs = self.window_sum_cs3D(IJ, self.win)

        win_size_cs = (self.win * 1.0) ** 3

        u_I_cs = I_sum_cs / win_size_cs
        u_J_cs = J_sum_cs / win_size_cs

        cross_cs = (
            IJ_sum_cs
            - u_J_cs * I_sum_cs
            - u_I_cs * J_sum_cs
            + u_I_cs * u_J_cs * win_size_cs
        )
        I_var_cs = I2_sum_cs - 2 * u_I_cs * I_sum_cs + u_I_cs * u_I_cs * win_size_cs
        J_var_cs = J2_sum_cs - 2 * u_J_cs * J_sum_cs + u_J_cs * u_J_cs * win_size_cs

        cc_cs = cross_cs * cross_cs / (I_var_cs * J_var_cs + self.eps)
        cc2 = cc_cs  # cross correlation squared

        # return negative cc.
        return 1.0 - torch.mean(cc2).float()
