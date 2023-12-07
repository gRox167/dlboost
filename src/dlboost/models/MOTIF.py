
from numpy import fix
import torch
from torch import nn
from torch.nn import functional as f

from einops import rearrange, repeat, reduce
import torchkbnufft as tkbn

from dlboost.models import SpatialTransformNetwork
from dlboost.utils.tensor_utils import interpolate


class CSE_ContraDyn_PhFix(nn.Module):
    def __init__(self, ch_pad, nufft_im_size, cse_module):
        super().__init__()
        # self.cse = cse_init 
        self.downsample = lambda x: interpolate(x, scale_factor=(1,0.5,0.5), mode='trilinear')
        self.upsample = lambda x: interpolate(x, scale_factor=(1,2,2), mode='trilinear')
        self.ch_pad= ch_pad
        self.cse_module = cse_module
        self.nufft_im_size = nufft_im_size
        self.nufft_adj = tkbn.KbNufftAdjoint(
            im_size=self.nufft_im_size)

    def kernel_estimate(self, kspace_data, kspace_traj):
        ph,ch,z,sp = kspace_data.shape
        image_init_ch = self.nufft_adj(rearrange(kspace_data, "ph ch z sp -> () (ch z) (ph sp)"),
                                       rearrange(kspace_traj, "ph comp sp -> comp (ph sp)"), 
                                       norm = 'ortho')
        image_init_ch = rearrange(image_init_ch, "() (ch z) h w -> ch z h w", ch=ch, z = z)
        ch,z,h,w = image_init_ch.shape
        image_init_ch_lr = image_init_ch.unsqueeze(0)
        for i in range(3):
            image_init_ch_lr = self.downsample(image_init_ch_lr)
        # image_init_ch_lr = interpolate(image_init_ch, scale_factor=0.25, mode='bilinear')
        if ch < self.ch_pad:
            image_init_ch_lr = f.pad(image_init_ch_lr, (0,0,0,0,0,0,0,self.ch_pad-ch))
        # print(image_init_ch_lr.shape)
        csm_lr = self.cse_module(image_init_ch_lr)
        csm_hr = csm_lr[:, :ch]
        for i in range(3):
            csm_hr = self.upsample(csm_hr)
        # csm_hr = self.upsample(csm_lr)
        # csm_hr = interpolate(csm_lr, scale_factor=4, mode='bilinear')
        # devide csm by its root square of sum
        csm_hr_norm = csm_hr / \
            torch.sqrt(torch.sum(torch.abs(csm_hr)**2, dim=1, keepdim=True))
        return repeat(csm_hr_norm, '() ch z h w -> ph ch z h w', ph=ph) 

    def forward(self, image, csm):
        # print(image.shape, csm.shape)
        ph, d, h, w = image.shape
        b, ch, d, h, w = csm.shape
        return image.unsqueeze(0,1)* csm.unsqueeze(2).expand(1,1,ph,1,1,1)

class MVF_Dyn(nn.Module):
    def __init__(self, size, regis_module):
        super().__init__()
        self.regis_module = regis_module
        self.spatial_transform = SpatialTransformNetwork(size=size, mode='bilinear')
        self.downsample = lambda x: interpolate(x, scale_factor=(1,0.5,0.5), mode='trilinear')
        self.upsample = lambda x: interpolate(x, scale_factor=(1,2,2), mode='trilinear')

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
        real  = self.spatial_transform(real, flow).squeeze((0,1))
        imag  = self.spatial_transform(imag, flow).squeeze((0,1))
        return torch.complex(real, imag)

    def mvf_composition(self, flow1, flow2):
        return flow1 + self.spatial_transform(flow2, flow1)
    
class MVF_Static(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.spatial_transform = SpatialTransformNetwork(size=size, mode='bilinear')

    def kernel_estimate(self, fixed, moving):
        return None
    
    def forward(self, moving, flow):
        real = moving.real[None, None, ...]
        imag = moving.imag[None, None, ...]
        real  = self.spatial_transform(real, flow).squeeze((0,1))
        imag  = self.spatial_transform(imag, flow).squeeze((0,1))
        return torch.complex(real, imag)

class NUFFT(nn.Module):
    def __init__(self, nufft_im_size):
        super().__init__()
        self.nufft_im_size = nufft_im_size
        self.nufft_op = tkbn.KbNufft(
            im_size=self.nufft_im_size)
        self.nufft_adj = tkbn.KbNufftAdjoint(
            im_size=self.nufft_im_size)
        
    def nufft_adj_forward(self, kspace_data, kspace_traj):
        ph, ch, z, sp = kspace_data.shape
        image = self.nufft_adj(
            rearrange(kspace_data, "ph ch z sp -> ph (ch z) sp"), 
            kspace_traj, norm = 'ortho')
        return rearrange(image, "ph (ch z) h w -> ph ch z h w", ch=ch, z = z)
    
    def nufft_forward(self, image, kspace_traj):
        ph, ch, z, h, w = image.shape
        # ph, ch, z, h, w = image.shape
        kspace_data = self.nufft_op(
            rearrange(image, "ph ch z h w -> ph (ch z) h w"), 
            kspace_traj, norm = 'ortho')
        return rearrange(kspace_data, "ph (ch z) sp -> ph ch z sp", ch=ch, z = z)

class MR_Forward_Model(nn.Module):
    def __init__(self, MVF_module: MVF_Dyn, CSE_module: CSE_ContraDyn_PhFix, NUFFT_module: NUFFT):
        super().__init__()
        self.MVF_module = MVF_module
        self.CSE_module = CSE_module
        self.NUFFT_module = NUFFT_module
        
    def forward(self, image, ref_list, kspace_traj, csm):
        image_list = []
        for i, ref in enumerate([image]+ref_list):
            if i != 0:
                flow = self.MVF_module.kernel_estimate(ref, image)
                moved = self.MVF_module(image, flow)
            else:
                moved = image
            image_list.append(moved)
        image_ph = torch.stack(image_list, dim=0)    
        # csm = self.CSE_module.kernel_estimate(image_ph)
        image_ch = self.CSE_module(image_ph, csm)
        kspace_data_estimated = self.NUFFT_module.nufft_forward(image_ch, kspace_traj)
        return kspace_data_estimated

class MOTIF_DataConsistency(nn.Module):
    def __init__(self, forward_model):
        super().__init__()
        self.forward_model = forward_model
        self.phase_num = 5
    
    def forward(self, x, csm, mvf):
        # first will be phase to be recon, other will be ref phase
        for moving_idx in range(self.phase_num):
            fix_indices = list(range(self.phase_num))
            fix_indices.pop(moving_idx)
            mvf_list = [mvf[:,i] for i in fix_indices]
            y = x.kspace_data
            image_ph = x.image[0]
            y_hat = self.forward_model(image_ph.requires_grad_(True), list(x.image[1:]), x.kspace_traj, csm)
            loss = 1/2 * torch.sum(torch.norm(y_hat-y)**2) # do i need to stop gradient here?
            gradient = torch.autograd.grad(loss, image_ph) #, create_graph=True)
        return gradient[0]

class MOTIF_Regularization(nn.Module):
    def __init__(self, recon_module):
        super().__init__()
        self.recon_module = recon_module
    
    def forward(self, x):
        return self.recon_module(x.image[None,None,...]).squeeze((0,1))

class MOTIF_Unrolling(nn.Module):
    def __init__(self, data_consistency_module, regularization_module, iterations, gamma_init=1.0, tau_init=1.0):
        super().__init__()
        self.data_consistency_module = data_consistency_module
        # self.regularization_module = regularization_module
        self.mvf_module = mvf_module
        self.csm_module = csm_module
        self.denoise_module = denoise_module
        self.iterations = iterations
        self.gamma = torch.nn.Parameter(torch.ones(self.iterations,dtype=torch.float32) * gamma_init)
        self.tau = torch.nn.Parameter(torch.ones(self.iterations,dtype = torch.float32) * tau_init)

    def forward(self, x, csm, mvf, kspace_data_csm, kspace_traj_csm):
        csm = self.csm_module.kernel_estimate(kspace_data_csm, kspace_traj_csm)
        mvf = None
        for t in range(self.iterations):
            x_dc, csm_dc, mvf_dc = self.data_consistency_module(x,csm,mvf)
            x_reg = self.denoise_module(x)
            csm_reg = self.csm_module(csm)
            mvf_reg = self.mvf_module(mvf)
            x = self.param_update_gd_step(x, x_dc, x_reg)
            csm = self.param_update_gd_step(csm, csm_dc, csm_reg)
            mvf = self.param_update_gd_step(mvf, mvf_dc, mvf_reg)
        return x

    def param_update_gd_step(self, x, dc, reg):
        return x - self.gamma * dc + self.tau * reg
        