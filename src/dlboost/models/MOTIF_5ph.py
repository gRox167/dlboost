import torch
import torchkbnufft as tkbn
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as f

from dlboost.models import SpatialTransformNetwork
from dlboost.utils.tensor_utils import interpolate


class CSE_ContraDyn_PhFix(nn.Module):
    def __init__(self, ch_pad, nufft_im_size, cse_module):
        super().__init__()
        # self.cse = cse_init
        self.downsample = lambda x: interpolate(
            x, scale_factor=(1, 0.5, 0.5), mode="trilinear"
        )
        self.upsample = lambda x: interpolate(
            x, scale_factor=(1, 2, 2), mode="trilinear"
        )
        self.ch_pad = ch_pad
        self.cse_module = cse_module
        self.nufft_im_size = nufft_im_size
        self.nufft_adj = tkbn.KbNufftAdjoint(im_size=self.nufft_im_size)

    def kernel_estimate(self, kspace_data, kspace_traj):
        ph, ch, z, sp = kspace_data.shape
        image_init_ch = self.nufft_adj(
            rearrange(kspace_data, "ph ch z sp -> () (ch z) (ph sp)"),
            rearrange(kspace_traj, "ph comp sp -> comp (ph sp)"),
            norm="ortho",
        )
        image_init_ch = rearrange(
            image_init_ch, "() (ch z) h w -> ch z h w", ch=ch, z=z
        )
        ch, z, h, w = image_init_ch.shape
        image_init_ch_lr = image_init_ch.unsqueeze(0)
        for i in range(3):
            image_init_ch_lr = self.downsample(image_init_ch_lr)
        # image_init_ch_lr = interpolate(image_init_ch, scale_factor=0.25, mode='bilinear')
        if ch < self.ch_pad:
            image_init_ch_lr = f.pad(
                image_init_ch_lr, (0, 0, 0, 0, 0, 0, 0, self.ch_pad - ch)
            )
        # print(image_init_ch_lr.shape)
        csm_lr = self.cse_module(image_init_ch_lr)
        csm_hr = csm_lr[:, :ch]
        for i in range(3):
            csm_hr = self.upsample(csm_hr)
        # csm_hr = self.upsample(csm_lr)
        # csm_hr = interpolate(csm_lr, scale_factor=4, mode='bilinear')
        # devide csm by its root square of sum
        csm_hr_norm = csm_hr / torch.sqrt(
            torch.sum(torch.abs(csm_hr) ** 2, dim=1, keepdim=True)
        )
        return repeat(csm_hr_norm, "() ch z h w -> ph ch z h w", ph=ph)

    def forward(self, image, csm):
        # print(image.shape, csm.shape)
        ph, d, h, w = image.shape
        b, ch, d, h, w = csm.shape
        return image.unsqueeze(0, 1) * csm.unsqueeze(2).expand(1, 1, ph, 1, 1, 1)


class MVF_Dyn(nn.Module):
    def __init__(self, size, regis_module):
        super().__init__()
        self.regis_module = regis_module
        self.spatial_transform = SpatialTransformNetwork(size=size, mode="bilinear")
        self.downsample = lambda x: interpolate(
            x, scale_factor=(1, 0.5, 0.5), mode="trilinear"
        )
        self.upsample = lambda x: interpolate(
            x, scale_factor=(1, 2, 2), mode="trilinear"
        )

    def kernel_estimate(self, fixed, moving):
        # input fixed and moving are complex images
        # input shape
        fixed_abs = self.downsample(fixed.abs()[None, None, ...])
        moving_abs = self.downsample(moving.abs()[None, None, ...])
        # print(fixed_abs.shape, moving_abs.shape)
        b, ch, z, h, w = fixed_abs.shape
        moved_abs, flow = self.regis_module(moving_abs, fixed_abs)
        return self.upsample(flow)

    def forward(self, moving, flow):
        real = moving.real[None, None, ...]
        imag = moving.imag[None, None, ...]
        real = self.spatial_transform(real, flow).squeeze((0, 1))
        imag = self.spatial_transform(imag, flow).squeeze((0, 1))
        return torch.complex(real, imag)


class MVF_Static(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.spatial_transform = SpatialTransformNetwork(size=size, mode="bilinear")

    def forward(self, moving, flow):
        real = moving.real[None, None, ...]
        imag = moving.imag[None, None, ...]
        # print(real.shape, flow.shape)
        real = self.spatial_transform(real, flow).squeeze((0, 1))
        imag = self.spatial_transform(imag, flow).squeeze((0, 1))
        return torch.complex(real, imag)


class NUFFT(nn.Module):
    def __init__(self, nufft_im_size):
        super().__init__()
        self.nufft_im_size = nufft_im_size
        self.nufft_op = tkbn.KbNufft(im_size=self.nufft_im_size)
        self.nufft_adj = tkbn.KbNufftAdjoint(im_size=self.nufft_im_size)

    def nufft_adj_forward(self, kspace_data, kspace_traj):
        ph, ch, z, sp = kspace_data.shape
        image = self.nufft_adj(
            rearrange(kspace_data, "ph ch z sp -> ph (ch z) sp"),
            kspace_traj,
            norm="ortho",
        )
        return rearrange(image, "ph (ch z) h w -> ph ch z h w", ch=ch, z=z)

    def nufft_forward(self, image, kspace_traj):
        ph, ch, z, h, w = image.shape
        # ph, ch, z, h, w = image.shape
        kspace_data = self.nufft_op(
            rearrange(image, "ph ch z h w -> ph (ch z) h w"), kspace_traj, norm="ortho"
        )
        return rearrange(kspace_data, "ph (ch z) sp -> ph ch z sp", ch=ch, z=z)


class MR_Forward_Model_Static(nn.Module):
    def __init__(self, mvf, csm, image_size, NUFFT_module: NUFFT):
        super().__init__()
        self.mvf = mvf
        self.csm = csm
        self.MVF_module = MVF_Static(size=image_size)
        self.NUFFT_module = NUFFT_module

    def forward(self, image, kspace_traj):
        image_list = []
        for i in range(5):
            if i != 0:
                # flow = self.MVF_module.kernel_estimate(ref, image)
                moved = self.MVF_module(image, self.mvf[i - 1])
            else:
                moved = image
            image_list.append(moved)
        image_ph = torch.stack(image_list, dim=0)
        # csm = self.CSE_module.kernel_estimate(image_ph)
        image_ch = image_ph.unsqueeze(1) * self.csm
        kspace_data_estimated = self.NUFFT_module.nufft_forward(image_ch, kspace_traj)
        return kspace_data_estimated, image_ph


class MOTIF_DataConsistency(nn.Module):
    def __init__(self, forward_model):
        super().__init__()
        self.forward_model = forward_model

    def forward(self, y_hat, y):
        # first will be phase to be recon, other will be ref phase
        loss = (
            1 / 2 * torch.sum(torch.norm(y_hat - y) ** 2)
        )  # do i need to stop gradient here?
        return gradient[0]


class MOTIF_Regularization(nn.Module):
    def __init__(self, recon_module):
        super().__init__()
        self.recon_module = recon_module

    def forward(self, x):
        return self.recon_module.predict_step(x.image[None, None, ...]).squeeze((0, 1))


class MOTIF_Unrolling(nn.Module):
    def __init__(
        self,
        data_consistency_module,
        regularization_module,
        iterations,
        gamma_init=1.0,
        tau_init=1.0,
    ):
        super().__init__()
        self.data_consistency_module = data_consistency_module
        self.regularization_module = regularization_module
        self.iterations = iterations
        self.gamma = torch.nn.Parameter(
            torch.ones(self.iterations, dtype=torch.float32) * gamma_init
        )
        self.tau = torch.nn.Parameter(
            torch.ones(self.iterations, dtype=torch.float32) * tau_init
        )

    def forward(self, x, kspace_data_csm, kspace_traj_csm, mvf):
        # y = x.clone()
        # estimate csm
        csm = self.data_consistency_module.forward_model.CSE_module.kernel_estimate(
            kspace_data_csm, kspace_traj_csm
        )
        for t in range(self.iterations):
            # x_ph_list = []
            # for i, x_ph in enumerate(x):
            #     x_ref = list(x)
            #     x_ref.pop(i)
            #     dc = self.data_consistency_module(MRData.stack([x_ph]+x_ref), csm)
            #     reg = self.regularization_module(x_ph)
            #     x_ph.image = x_ph.image - self.gamma[t] * dc + self.tau[t] * reg
            #     x_ph_list.append(x_ph)
            # x = MRData.stack(x_ph_list)
            dc = self.data_consistency_module(x, csm)
            reg = self.regularization_module(x)  # and update csm
            x.image = x.image - self.gamma * dc + self.tau * reg
        return x
