import lightning.pytorch as pl
import torch
import torchkbnufft as tkbn
import torchopt
from dlboost.models import SpatialTransformNetwork
from dlboost.tasks.boilerplate_MOTIF import MOTIF
from dlboost.utils.tensor_utils import interpolate
from einops import rearrange, repeat
from torch import nn, optim
from torch.nn import functional as f
from torchopt import pytree


class Recon(MOTIF):
    def __init__(
        self,
        recon_module: nn.Module,
        cse_module: nn.Module,
        regis_module: nn.Module,
        nufft_im_size=(320, 320),
        patch_size=(64, 64),
        ch_pad=42,
        recon_loss_fn=nn.MSELoss,
        smooth_loss_coef=0.1,
        recon_optimizer=optim.Adam,
        recon_lr=1e-4,
        **kwargs,
    ):
        super().__init__(
            recon_module,
            cse_module,
            regis_module,
            nufft_im_size,
            patch_size,
            ch_pad,
            recon_loss_fn,
            smooth_loss_coef,
            recon_optimizer,
            recon_lr,
            **kwargs,
        )

    def training_step(self, batch, batch_idx):
        recon_opt = self.optimizers()
        recon_opt.zero_grad()
        batch = batch[0]

        kspace_data, kspace_traj = batch["kspace_data_z"], batch["kspace_traj"]
        kspace_data_compensated = batch["kspace_data_z_compensated"]
        kspace_data_cse, kspace_traj_cse = (
            batch["kspace_data_z_cse"],
            batch["kspace_traj_cse"],
        )

        # currently the shape of the kspace_data is (b, ph, ch, z, sp), randomly select a phase
        # and then select the corresponding kspace_data and kspace_traj as reference data,
        # then select the remaining kspace_data and kspace_traj as source data
        ref_idx = torch.randint(0, 5, (1,))
        kspace_data_ref = kspace_data[:, ref_idx]
        kspace_traj_ref = kspace_traj[:, ref_idx]
        kspace_data_src = torch.cat(
            (kspace_data[:, :ref_idx], kspace_data[:, ref_idx + 1 :]), dim=1
        )
        kspace_traj_src = torch.cat(
            (kspace_traj[:, :ref_idx], kspace_traj[:, ref_idx + 1 :]), dim=1
        )

        csm_init = self.nufft_adj_forward(kspace_data_cse, kspace_traj_cse)
        image = self.nufft_adj_forward(kspace_data_compensated, kspace_traj)
        csm = self.csm_module.kernel_estimate(csm_init)
        image = torch.sum(image * csm.conj(), dim=1)
        mvf = torch.zeros_like(image).expand(-1, 3, -1, -1, -1)


class CSE_DynPh(nn.Module):
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
    

    def kernel_regularize(self, csm_kernel):
        ph,ch,d,h,w = csm_kernel.shape
        if ch < self.ch_pad:
            _csm = f.pad(
                csm_kernel, (0, 0, 0, 0, 0, 0, 0, self.ch_pad - ch)
            )
        else:
            raise ValueError("ch_pad should be larger or equal to coil channel number")
        _csm = self.cse_module(_csm)[:, :ch]
        _csm = _csm / torch.sqrt(
            torch.sum(torch.abs(_csm) ** 2, dim=1, keepdim=True)
        )
        return _csm

    def forward(self, image, csm_kernel):
        # print(image.shape, csm.shape)
        ph, d, h, w = image.shape
        _csm = csm_kernel
        for i in range(2):
            _csm = self.upsample(_csm)
        ph, ch, d, h, w = _csm.shape
        return image.unsqueeze(0, 1) * _csm.unsqueeze(2).expand(1, 1, ph, 1, 1, 1)


class MVF_Dyn(nn.Module):
    def __init__(self, size, regis_module,ph_num=5):
        super().__init__()
        self.regis_module = regis_module
        self.spatial_transform = SpatialTransformNetwork(size=size, mode="bilinear")
        self.downsample = lambda x: interpolate(
            x, scale_factor=(1, 0.5, 0.5), mode="trilinear"
        )
        self.upsample = lambda x: interpolate(
            x, scale_factor=(1, 2, 2), mode="trilinear"
        )
        
    
    def kernel_regularize(self, mvf_kernels):
        _mvf_kernels = [self.regularize_module(mvf) for mvf in mvf_kernels]
        return _mvf

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
        # self.MVF_module = (size=image_size)
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


class MOTIF_Regularization(nn.Module):
    def __init__(self, denoise_module, mvf_module, csm_module):
        super().__init__()
        self.denoise_module = denoise_module
        self.mvf_module = mvf_module
        self.csm_module = csm_module

    def forward(self, x):
        return self.recon_module.predict_step(x.image[None, None, ...]).squeeze((0, 1))


class MOTIF_Unrolling(nn.Module):
    def __init__(
        self,
        forward_model,
        regularization_module,
        iterations,
        gamma_init=1.0,
        tau_init=1.0,
    ):
        super().__init__()
        self.forward_model = forward_model
        self.regularization_module = regularization_module
        self.iterations = iterations
        self.gamma = torch.nn.Parameter(
            torch.ones(self.iterations, dtype=torch.float32) * gamma_init
        )
        self.tau = torch.nn.Parameter(
            torch.ones(self.iterations, dtype=torch.float32) * tau_init
        )
        self.downsample = lambda x: interpolate(
            x, scale_factor=(1, 0.5, 0.5), mode="trilinear"
        )

    def forward(self, kspace_data, kspace_traj):
        # initialization
        image_multi_ch = self.nufft_adj_forward(kspace_data, kspace_traj)
        image_init = self.image_init(image_multi_ch)
        params = {"image": image_init, 
                  "csm": self.cse_kernel_init(image_multi_ch), 
                  "mvf": self.mvf_kernel_init(image_init)}

        for t in range(self.iterations):
            # apply forward model to get kspace_data_estimated
            kspace_data_estimated = self.forward_model(params, kspace_traj)
            loss = (
                1 / 2 * torch.sum(torch.norm(kspace_data_estimated - kspace_data) ** 2)
            )
            updates = self.update(
                torch.autograd.grad(loss, params), self.regularization_module(params)
            )
            params = torchopt.apply_updates(params, updates)
        return params

    def update(self, dc_grads, reg_grads):
        updates = pytree.tree_map(
            lambda dc_grad, reg_grad: -self.gamma * dc_grad + self.tau * reg_grad,
            dc_grads,
            reg_grads,
        )
        return updates
    
    def image_init(self, image_multi_ch):
        image_init = torch.sum(image_multi_ch * image_multi_ch.conj(), dim=1, keepdim=True)
        return image_init

    def cse_kernel_init(self, image_multi_ch):
        cse_kernel = image_multi_ch.clone()
        for i in range(2):
            cse_kernel = self.downsample(cse_kernel)
        return cse_kernel

    def mvf_kernel_init(self, image):
        for i in range(2):
            _image = self.downsample(image)
        mvf_kernels= [torch.zeros_like(_image).expand(-1, 3, -1, -1, -1) for i in range(self.ph_num-1)]
        return mvf_kernels

    # def nufft_forward(self, image_init, kspace_traj):
    #     ph, ch, d, h, w = image_init.shape
    #     kspace_data = self.nufft_op(
    #         rearrange(image_init, 'ph ch d h w -> ph (ch d) h w'),
    #         kspace_traj, norm='ortho')
    #     return rearrange(kspace_data, 'ph (ch d) len -> ph ch d len', ch=ch)

    def nufft_adj_forward(self, kspace_data, kspace_traj):
        b, ph, ch, d, length = kspace_data.shape
        # breakpoint()
        image = self.nufft_adj(
            rearrange(kspace_data, 'b ph ch d len -> b (ch d) (ph len)'),
            rearrange(kspace_traj, 'b ph comp len -> b comp (ph len)'), norm='ortho')
        return rearrange(image, 'b (ch d) h w -> b ch d h w', ch=ch)