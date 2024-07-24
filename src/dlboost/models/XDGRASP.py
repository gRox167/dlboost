import einx
import torch

# from dlboost.utils.tensor_utils import formap, interpolate
from torch import nn
from torchmin import minimize

from mrboost.computation import nufft_2d, nufft_adj_2d


class CSE_Static(nn.Module):
    def __init__(self):
        super().__init__()

    def generate_forward_operator(self, csm_kernels):
        self._csm = csm_kernels
        # self._csm = _csm / \
        #     torch.sqrt(torch.sum(torch.abs(_csm)**2, dim=2, keepdim=True))

    def forward(self, image):
        if len(image.shape) == 5:
            return einx.dot("t ph d h w, t ch d h w -> t ph ch d h w", image, self._csm)
        elif len(image.shape) == 4:
            return einx.dot("t d h w, t ch d h w -> t ch d h w", image, self._csm)
        # return einx.dot("t ph d h w, t ch d h w -> t ph ch d h w", image, self._csm)

        # return image * self._csm.unsqueeze(1).expand_as(image)


class NUFFT(nn.Module):
    def __init__(self, nufft_im_size):
        super().__init__()
        self.nufft_im_size = nufft_im_size

    def generate_forward_operator(self, kspace_traj):
        self.kspace_traj = kspace_traj

    def adjoint(self, kspace_data):
        return nufft_adj_2d(kspace_data, self.kspace_traj, self.nufft_im_size)

    def forward(self, image):
        return nufft_2d(image, self.kspace_traj, self.nufft_im_size)


class MR_Forward_Model_Static(nn.Module):
    def __init__(self, image_size, nufft_im_size):
        super().__init__()
        self.S = CSE_Static()
        self.N = NUFFT(nufft_im_size)

    def generate_forward_operators(self, csm_kernels, kspace_traj):
        self.S.generate_forward_operator(csm_kernels)
        self.N.generate_forward_operator(kspace_traj)

    def forward(self, params):
        image = params
        image_multi_ch = self.S(image)
        kspace_data_estimated = self.N(image_multi_ch)
        return kspace_data_estimated


class RespiratoryTVRegularization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, params):
        # compute the respiratory tv regularization
        if len(params.shape) == 5:
            diff = params[:, 1:] - params[:, :-1]
            return diff.abs().mean()
        elif len(params.shape) == 4:
            return 0


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
        self.nufft_im_size = nufft_im_size

    def forward(self, kspace_data, kspace_data_compensated, kspace_traj, csm):
        # initialization
        csm_ = csm / torch.sqrt(torch.sum(torch.abs(csm) ** 2, dim=2, keepdim=True))
        if self.lambda2 == 0:
            kspace_data = einx.rearrange("t ph ch z k -> t ch z (ph k)", kspace_data)
            kspace_data_compensated = einx.rearrange(
                "t ph ch z k -> t ch z (ph k)", kspace_data_compensated
            )
            kspace_traj = einx.rearrange("t ph c k -> t c (ph k)", kspace_traj)
        ic(
            kspace_data.shape,
            kspace_data_compensated.shape,
            kspace_traj.shape,
            csm_.shape,
        )
        image_init = (
            self.generate_init_image(kspace_data_compensated, kspace_traj, csm_) / 5
        )
        ic(image_init.shape)
        # image_init = torch.zeros_like(image_init)
        params = image_init.clone().requires_grad_(True)

        def fun(params):
            _params = torch.view_as_complex(params)
            self.forward_model.generate_forward_operators(csm_, kspace_traj)
            kspace_data_estimated = self.forward_model(_params)
            # dc_diff = kspace_data_estimated - kspace_data
            # loss_dc = 1 / 2 * (dc_diff.abs() ** 2).mean()
            # x * x.conj() = x.abs()**2 in theory, however, in practice the LHS is a complex number with a very small imaginary part
            loss_dc = (
                1
                / 2
                * torch.mean(
                    (
                        torch.view_as_real(kspace_data_estimated)
                        - torch.view_as_real(kspace_data)
                    )
                    ** 2
                )
            )
            loss_reg = self.lambda1 * self.contrast_TV(
                _params
            ) + self.lambda2 * self.respiratory_TV(_params)
            print(loss_dc.item(), loss_reg.item())
            return loss_dc + loss_reg
            # return loss_dc

        result = minimize(
            fun, torch.view_as_real(params), method="CG", tol=1e-6, disp=1
        )
        return torch.view_as_complex(result.x), image_init
        # result = torch.zeros_like(params)
        # return result, image_init

    def generate_init_image(self, kspace_data, kspace_traj, csm):
        image_init = nufft_adj_2d(kspace_data, kspace_traj, self.nufft_im_size)
        ic(image_init.shape)
        if len(image_init.shape) == 6:
            return einx.dot(
                "t ph ch d h w, t ch d h w -> t ph d h w", image_init, csm.conj()
            )
        elif len(image_init.shape) == 5:
            return einx.dot("t ch d h w, t ch d h w -> t d h w", image_init, csm.conj())
        # return torch.sum(image_init*csm.unsqueeze(1).expand_as(image_init).conj(), dim=2)


# class NUFFT(nn.Module):
#     def __init__(self, nufft_im_size):
#         super().__init__()
#         self.nufft_im_size = nufft_im_size
#         self.nufft = tkbn.KbNufft(im_size=self.nufft_im_size)
#         self.nufft_adj = tkbn.KbNufftAdjoint(im_size=self.nufft_im_size)

#     def generate_forward_operator(self, kspace_traj):
#         self.kspace_traj = kspace_traj

#     def adjoint(self, kspace_data):
#         b, ph, ch, d, sp = kspace_data.shape
#         images = []
#         for k, kj in zip(
#             torch.unbind(kspace_data, dim=1), torch.unbind(self.kspace_traj, dim=1)
#         ):
#             image = self.nufft_adj(
#                 rearrange(k, "b ch d len -> b (ch d) len"), kj, norm="ortho"
#             )
#             images.append(
#                 rearrange(image, "b (ch d) h w -> b ch d h w", b=b, ch=ch, d=d)
#             )
#         return torch.stack(images, dim=1)

#     def forward(self, image):
#         b, ph, ch, d, h, w = image.shape
#         # b, ph, comp, sp = self.kspace_traj.shape
#         kspace_data_list = []
#         for i, kj in zip(
#             torch.unbind(image, dim=1), torch.unbind(self.kspace_traj, dim=1)
#         ):
#             kspace_data = self.nufft(
#                 rearrange(i, "b ch d h w -> b (ch d) h w"),
#                 kj,
#                 norm="ortho",
#             )
#             kspace_data_list.append(
#                 rearrange(kspace_data, "b (ch d) len -> b ch d len", ch=ch, d=d)
#             )
#         return torch.stack(kspace_data_list, dim=1)
