import torch
from torch import nn
import torchkbnufft as tkbn

# from dlboost.utils.tensor_utils import formap, interpolate
from einops import rearrange, einsum
from torchmin import minimize


class CSE_Static(nn.Module):
    def __init__(self):
        super().__init__()

    def generate_forward_operator(self, csm_kernels):
        self._csm = csm_kernels
        # self._csm = _csm / \
        #     torch.sqrt(torch.sum(torch.abs(_csm)**2, dim=2, keepdim=True))

    def forward(self, image):
        # print(image.shape, self._csm.shape)
        return einsum(image, self._csm, "t ph d h w, t ch d h w -> t ph ch d h w")

        # return image * self._csm.unsqueeze(1).expand_as(image)



class NUFFT(nn.Module):
    def __init__(self, nufft_im_size):
        super().__init__()
        self.nufft_im_size = nufft_im_size
        self.nufft = tkbn.KbNufft(im_size=self.nufft_im_size)
        self.nufft_adj = tkbn.KbNufftAdjoint(im_size=self.nufft_im_size)

    def generate_forward_operator(self, kspace_traj):
        self.kspace_traj = kspace_traj

    def adjoint(self, kspace_data):
        b, ph, ch, d, sp = kspace_data.shape
        images = []
        for k, kj in zip(
            torch.unbind(kspace_data, dim=1), torch.unbind(self.kspace_traj, dim=1)
        ):
            image = self.nufft_adj(
                rearrange(k, "b ch d len -> b (ch d) len"), kj, norm="ortho"
            )
            images.append(
                rearrange(image, "b (ch d) h w -> b ch d h w", b=b, ch=ch, d=d)
            )
        return torch.stack(images, dim=1)

    def forward(self, image):
        b, ph, ch, d, h, w = image.shape
        # b, ph, comp, sp = self.kspace_traj.shape
        kspace_data_list = []
        for i, kj in zip(
            torch.unbind(image, dim=1), torch.unbind(self.kspace_traj, dim=1)
        ):
            kspace_data = self.nufft(
                rearrange(i, "b ch d h w -> b (ch d) h w"),
                kj,
                norm="ortho",
            )
            kspace_data_list.append(
                rearrange(kspace_data, "b (ch d) len -> b ch d len", ch=ch, d=d)
            )
        return torch.stack(kspace_data_list, dim=1)


class MR_Forward_Model_Static(nn.Module):
    def __init__(self, image_size, nufft_im_size):
        super().__init__()
        self.S = CSE_Static()
        self.N = NUFFT(nufft_im_size)

    def generate_forward_operators(self, csm_kernels, kspace_traj):
        self.S.generate_forward_operator(csm_kernels)
        self.N.generate_forward_operator(kspace_traj)

    def forward(self, params):
        image_5ph = params
        # image_5ph = image.expand(-1, 5, -1, -1, -1)
        image_5ph_multi_ch = self.S(image_5ph)
        # kspace_data_estimated = self.N(image)
        kspace_data_estimated = self.N(image_5ph_multi_ch)
        return kspace_data_estimated


class RespiratoryTVRegularization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, params):
        # compute the respiratory tv regularization
        diff = params[:, 1:] - params[:, :-1]
        return diff.abs().mean()

class ContrastTVRegularization(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, params):
        # compute the contrast tv regularization
        diff = params[1:] - params[:-1]
        return diff.abs().mean()

class Identity_Regularization:
    def __init__(self):
        pass

    def __call__(self, params):
        return params


class XDGRASP(nn.Module):
    def __init__(
        self,
        patch_size,
        nufft_im_size,
        lambda1,
        lambda2,
    ):
        super().__init__()
        self.forward_model = MR_Forward_Model_Static(patch_size, nufft_im_size)
        self.contrast_TV = ContrastTVRegularization()
        self.respiratory_TV = RespiratoryTVRegularization()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.nufft_adj = tkbn.KbNufftAdjoint(im_size=nufft_im_size)

    def forward(self, kspace_data, kspace_data_compensated, kspace_traj, csm):
        # initialization
        csm_ = csm / \
            torch.sqrt(torch.sum(torch.abs(csm)**2, dim=2, keepdim=True))
        image_init = self.nufft_adjoint(kspace_data_compensated, kspace_traj, csm_)/5
        print(image_init)
        # image_init = torch.zeros_like(image_init)
        params = image_init.clone().requires_grad_(True)
        def fun(params):
            _params = torch.view_as_complex(params)
            self.forward_model.generate_forward_operators(csm_, kspace_traj)
            kspace_data_estimated = self.forward_model(_params)
            dc_diff = kspace_data_estimated - kspace_data
            loss_dc = 1/2 * (dc_diff.abs()**2).mean()
            # x * x.conj() = x.abs()**2 in theory, however, in practice the LHS is a complex number with a very small imaginary part
            # loss_dc = 1 / 2 * torch.norm(kspace_data_estimated - kspace_data, 2)
            loss_reg = self.lambda1 * self.contrast_TV(_params) + self.lambda2 * self.respiratory_TV(_params)
            print(loss_dc.item(), loss_reg.item())
            return loss_dc + loss_reg
        result = minimize(fun, torch.view_as_real(params), method='CG', tol=1e-6, disp=1)
        return torch.view_as_complex(result.x), image_init
        # result = torch.zeros_like(params)
        # return result, image_init
    
    def nufft_adjoint(self, kspace_data, kspace_traj, csm):
        images=[]
        b, ph, ch, d, sp = kspace_data.shape
        for k,kj in zip(torch.unbind(kspace_data, 0), torch.unbind(kspace_traj, 0)):
            image = self.nufft_adj(
                rearrange(k, "ph ch d len -> ph (ch d) len"), kj, norm="ortho"
            )
            images.append(
                rearrange(image, "ph (ch d) h w -> ph ch d h w", ch=ch, d=d)
            )
        image_init = torch.stack(images, dim=0)
        return einsum(image_init, csm.conj(), "t ph ch d h w, t ch d h w -> t ph d h w")
        # return torch.sum(image_init*csm.unsqueeze(1).expand_as(image_init).conj(), dim=2)
     