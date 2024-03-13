from copy import deepcopy

import torch
import torchkbnufft as tkbn
import torchopt
from einops import rearrange, reduce, repeat
from torch import nn, vmap
from torch.nn import functional as f
from torchopt import pytree

from dlboost.models import ComplexUnet, DWUNet, SpatialTransformNetwork
from dlboost.utils.tensor_utils import interpolate


class CSM_DynPh(nn.Module):
    def __init__(self, upsample_times=1):
        super().__init__()
        if upsample_times:

            def upsample(x):
                for i in range(upsample_times):
                    x = interpolate(x, scale_factor=(1, 2, 2), mode="trilinear")
                return x
        else:
            upsample = lambda x: x  # noqa: E731
        self.upsample = vmap(upsample)

    def generate_forward_operator(self, csm_kernels):
        _csm = self.upsample(csm_kernels.clone())
        self._csm = _csm / torch.sqrt(
            torch.sum(torch.abs(_csm) ** 2, dim=2, keepdim=True)
        )

    def forward(self, image):
        return image.unsqueeze(2) * self._csm


class MVF_Dyn(nn.Module):
    def __init__(self, size, upsample_times=1):
        super().__init__()
        self.spatial_transform = SpatialTransformNetwork(size=size, mode="bilinear")
        # self.downsample = lambda x: interpolate(
        #     x, scale_factor=(1, 0.5, 0.5), mode="trilinear"
        # )
        if upsample_times:

            def upsample(x):
                for i in range(upsample_times):
                    x = interpolate(x, scale_factor=(1, 2, 2), mode="trilinear")
                return x
        else:
            upsample = lambda x: x  # noqa: E731
        self.upsample = upsample
        # self.upsample_times = upsample_times

    def generate_forward_operator(self, mvf_kernels):
        self.ph_to_move = mvf_kernels.shape[1]
        _mvf = rearrange(mvf_kernels.clone(), "b ph v d h w -> (b ph) v d h w")
        self._mvf = self.upsample(_mvf)

    def forward(self, image):
        # image is a tensor with shape b, 1, d, h, w, 2
        # image_ref = image.clone()
        image_move = repeat(
            torch.view_as_real(image),
            "b () d h w comp -> (b ph) comp d h w",
            ph=self.ph_to_move,
        )
        # rearrange the image to (b, ph), 2, d, h, w
        image_4ph = self.spatial_transform(image_move, self._mvf)
        image_4ph = rearrange(
            image_4ph, "(b ph) comp d h w -> b ph d h w comp", ph=self.ph_to_move
        )
        image_4ph = torch.complex(image_4ph[..., 0], image_4ph[..., 1])
        return torch.cat((image, image_4ph), dim=1)


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
    def __init__(
        self,
        image_size,
        nufft_im_size,
        csm_updample_times,
        mvf_upsample_times,
        CSM_module: CSM_DynPh,
        MVF_module: MVF_Dyn,
        NUFFT_module: NUFFT,
    ):
        super().__init__()
        self.M = MVF_module(image_size, mvf_upsample_times) if MVF_module else None
        self.S = CSM_module(csm_updample_times)
        self.N = NUFFT_module(nufft_im_size)

    def generate_forward_operators(self, mvf_kernels, csm_kernels, kspace_traj):
        self.M.generate_forward_operator(mvf_kernels) if self.M else None
        self.S.generate_forward_operator(csm_kernels)
        self.N.generate_forward_operator(kspace_traj)

    def forward(self, image):
        _image = image.clone()
        image_5ph = self.M(_image) if self.M else _image.expand(-1, 5, -1, -1, -1)
        image_5ph_multi_ch = self.S(image_5ph)
        # kspace_data_estimated = self.N(image)
        kspace_data_estimated = self.N(image_5ph_multi_ch)
        return kspace_data_estimated


""" class CG_DC(nn.Module):
    def __init__(self, forward_model):
        self.forward_model = forward_model
    
    def forward(self, params, y):
        def fun(params):
            return 1/2 * torch.sum((self.forward_model(params)-y)**2)
        return minimize(fun, params, method='cg').x """


class Regularization(nn.Module):
    def __init__(self, ch_pad):
        super().__init__()
        self.ch_pad = ch_pad
        self.image_denoiser = ComplexUnet(
            1,
            1,
            spatial_dims=3,
            conv_net=DWUNet(
                in_channels=2,
                out_channels=2,
                # features=(16, 32, 64, 128, 256),
                features=(32, 64, 128, 256, 512),
                strides=((2, 4, 4), (2, 2, 2), (1, 2, 2), (1, 2, 2)),
                kernel_sizes=(
                    (3, 7, 7),
                    (3, 7, 7),
                    (3, 7, 7),
                    (3, 7, 7),
                    (3, 7, 7),
                ),
            ),
        )
        self.csm_denoiser = ComplexUnet(
            in_channels=ch_pad,
            out_channels=ch_pad,
            spatial_dims=3,
            conv_net=DWUNet(
                in_channels=2 * ch_pad,
                out_channels=2 * ch_pad,
                spatial_dims=3,
                strides=((2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 1, 1)),
                kernel_sizes=(
                    (3, 7, 7),
                    (3, 7, 7),
                    (3, 7, 7),
                    (3, 7, 7),
                    (3, 7, 7),
                ),
                features=(32, 64, 128, 128, 128),
            ),
        )
        self.mvf_denoiser = DWUNet(
            in_channels=3,
            out_channels=3,
            spatial_dims=3,
            strides=((2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 1, 1)),
            kernel_sizes=(
                (3, 7, 7),
                (3, 7, 7),
                (3, 7, 7),
                (3, 7, 7),
                (3, 7, 7),
            ),
            features=(8, 16, 32, 64, 128),
        )

    def forward(self, params):
        return {
            "image": self.image_denoiser_forward(params["image"]),
            "csm": self.csm_denoiser_forward(params["csm"]),
            "mvf": self.mvf_denoiser_forward(params["mvf"]),
        }

    def image_denoiser_forward(self, image):
        result = self.image_denoiser(image)
        # breakpoint()
        return result

    def csm_denoiser_forward(self, csm_kernel):
        return csm_kernel
        b, ph, ch, d, h, w = csm_kernel.shape
        if ch <= self.ch_pad:
            _csm = f.pad(csm_kernel, (0, 0, 0, 0, 0, 0, 0, self.ch_pad - ch))
        else:
            raise ValueError("ch_pad should be larger or equal to coil channel number")
        _csm = rearrange(_csm, "b ph ch d h w -> (b ph) ch d h w")
        _csm = self.csm_denoiser(_csm)[:, :ch]
        # _csm = _csm / torch.sqrt(torch.sum(torch.abs(_csm) ** 2, dim=1, keepdim=True))
        result = rearrange(_csm, "(b ph) ch d h w -> b ph ch d h w", ph=ph)
        return result

    def mvf_denoiser_forward(self, mvf_kernels):
        return mvf_kernels
        result = torch.stack(
            [self.mvf_denoiser(mvf) for mvf in torch.unbind(mvf_kernels, dim=1)], dim=1
        )
        return result


class Identity_Regularization:
    def __init__(self):
        pass

    def __call__(self, params):
        return params


class SD_RED(nn.Module):
    def __init__(
        self,
        # forward_model,
        patch_size,
        nufft_im_size,
        ch_pad,
        iterations,
        gamma_init=0.0001,
        tau_init=0.2,
    ):
        super().__init__()
        self.forward_model = MR_Forward_Model_Static(patch_size, nufft_im_size)
        self.regularization = Regularization(ch_pad)
        # self.regularization = Identity_Regularization()
        self.iterations = iterations
        # self.gamma = torch.nn.Parameter(
        #     torch.ones(self.iterations, dtype=torch.float32) * gamma_init
        # )
        # self.tau = torch.nn.Parameter(
        #     torch.ones(self.iterations, dtype=torch.float32) * tau_init
        # )
        self.gamma = gamma_init
        self.tau = tau_init
        self.downsample = lambda x: interpolate(
            x, scale_factor=(1, 0.5, 0.5), mode="trilinear"
        )
        self.nufft_adj = tkbn.KbNufftAdjoint(im_size=nufft_im_size)

    def forward(self, kspace_data, kspace_traj, weights):
        # initialization
        image_multi_ch = self.nufft_adjoint(kspace_data, kspace_traj)
        self.ph_num = image_multi_ch.shape[1]
        image_list = []
        csm_init = self.csm_kernel_init(image_multi_ch)
        image_init = self.image_init(image_multi_ch, csm_init)
        image_list.append(image_init.clone().detach().cpu())
        mvf_init = self.mvf_kernel_init(image_init)
        params = {
            "image": image_init.requires_grad_(True),
            "csm": csm_init.requires_grad_(True),
            "mvf": mvf_init.requires_grad_(True),
        }
        # state = opt.init(params)

        for t in range(self.iterations):
            # apply forward model to get kspace_data_estimated
            self.forward_model.generate_forward_operators(
                params["mvf"], params["csm"], kspace_traj
            )
            kspace_data_estimated = self.forward_model(params["image"])
            # loss_dc = 1 / 2 * torch.norm(kspace_data_estimated - kspace_data, 1)
            loss_dc = (
                1
                / 2
                * (
                    torch.abs(
                        torch.view_as_real(kspace_data_estimated)
                        - torch.view_as_real(kspace_data)
                    )
                    ** 2
                ).sum()
            )
            grad_dc = torch.autograd.grad(
                loss_dc, (params["image"], params["csm"], params["mvf"])
            )
            grad_dc = {
                # "image": 0,
                # "csm": 0,
                # "mvf": 0,
                "image": grad_dc[0],
                "csm": grad_dc[1],
                "mvf": grad_dc[2],
            }
            # loss_reg = self.regularization(params)
            # grad_reg = pytree.tree_map(lambda x, reg: (x - reg), params, loss_reg)
            grad_reg = pytree.tree_map(lambda x: 0, params)
            updates = self.update(grad_dc, grad_reg, t)
            params = self.apply_updates(params, updates)
            image_list.append(params["image"].clone().detach().cpu())
            print(f"t: {t}, loss: {loss_dc}")
        # print(f"gamma: {self.gamma}, tau: {self.tau[0]}")
        return params, image_list

    def update(self, dc_grads, reg_grads, t):
        return pytree.tree_map(
            lambda dc_grad, reg_grad: -(self.gamma * dc_grad + self.tau * reg_grad),
            dc_grads,
            reg_grads,
        )

    def apply_updates(self, params, updates):
        return pytree.tree_map(lambda param, update: param + update, params, updates)

    def image_init(self, image_multi_ch, csm_init):
        # image_init = torch.sum(image_multi_ch * image_multi_ch.conj(), dim=2).mean(dim=1, keepdim=True)**0.5
        ph1 = image_multi_ch[:, 0:1]
        # ph1_conj = ph1.conj()
        csm = ph1 / torch.sqrt(torch.sum(torch.abs(ph1) ** 2, dim=1, keepdim=True))
        image_init = torch.sum(ph1 * csm.conj(), dim=2) ** 0.5
        return image_init

    def csm_kernel_init(self, image_multi_ch):
        b, ph, _, _, _, _ = image_multi_ch.shape
        csm_kernel = rearrange(image_multi_ch, "b ph ch d h w -> (b ph) ch d h w")
        csm_kernel = csm_kernel / torch.sqrt(
            torch.sum(torch.abs(csm_kernel) ** 2, dim=1, keepdim=True)
        )
        for i in range(3):
            csm_kernel = self.downsample(csm_kernel)
        return rearrange(csm_kernel, "(b ph) ch d h w -> b ph ch d h w", b=b, ph=ph)

    def mvf_kernel_init(self, image):
        b, _, d, h, w = image.shape
        mvf_kernels = [
            torch.zeros((b, 3, d, h // 2, w // 2), device=image.device)
            for i in range(self.ph_num - 1)
        ]
        return torch.stack(mvf_kernels, dim=1)

    def nufft_adjoint(self, kspace_data, kspace_traj):
        b, ph, ch, d, length = kspace_data.shape
        images = []
        for k, kj in zip(
            torch.unbind(kspace_data, dim=1), torch.unbind(kspace_traj, dim=1)
        ):
            image = self.nufft_adj(
                rearrange(k, "b ch d len -> b (ch d) len"), kj, norm="ortho"
            )
            images.append(
                rearrange(image, "b (ch d) h w -> b ch d h w", b=b, ch=ch, d=d)
            )
        return torch.stack(images, dim=1)


class ADAM_RED(SD_RED):
    def __init__(
        self,
        # forward_model,
        patch_size,
        nufft_im_size,
        ch_pad,
        iterations,
        gamma_init=0.1,
        tau_init=0.33333,
    ):
        super().__init__(
            patch_size,
            nufft_im_size,
            ch_pad,
            iterations,
            gamma_init,
            tau_init,
        )

    def forward(self, kspace_data, kspace_traj, weights):
        opt = torchopt.adam(self.gamma[0])
        # initialization
        image_multi_ch = self.nufft_adjoint(kspace_data, kspace_traj)
        self.ph_num = image_multi_ch.shape[1]
        image_init = self.image_init(image_multi_ch)
        image_init_output = image_init.clone().detach().cpu()
        csm_init = self.csm_kernel_init(image_multi_ch)
        mvf_init = self.mvf_kernel_init(image_init)
        params = {
            "image": image_init.requires_grad_(True),
            "csm": csm_init.requires_grad_(True),
            "mvf": mvf_init.requires_grad_(True),
        }
        state = opt.init(params)

        for t in range(self.iterations):
            # apply forward model to get kspace_data_estimated
            self.forward_model.generate_forward_operators(
                params["mvf"], params["csm"], kspace_traj
            )
            kspace_data_estimated = self.forward_model(params)
            loss_dc = (
                1
                / 2
                * (
                    torch.abs(
                        torch.view_as_real(kspace_data_estimated)
                        - torch.view_as_real(kspace_data)
                    )
                ).mean()
            )
            # grad_dc = torch.autograd.grad(loss_dc, params["csm"],)
            grad_dc = torch.autograd.grad(
                # loss_dc, (params["image"], params["csm"])
                loss_dc,
                (params["image"], params["csm"], params["mvf"]),
            )
            grad_dc = {
                "image": grad_dc[0],
                "csm": grad_dc[1],
                "mvf": grad_dc[2],
                # "mvf": torch.zeros_like(params["mvf"]),
            }
            loss_reg = self.regularization(params)
            grad_reg = pytree.tree_map(lambda x, reg: x - reg, params, loss_reg)
            updates, state = opt.update(grad_dc, state, inplace=False)
            updates = pytree.tree_map(
                lambda dc, reg: dc - self.tau[t] * reg, updates, grad_reg
            )
            params = torchopt.apply_updates(params, updates)
            print(f"iteration {t}; loss_dc: {loss_dc};")
        return params, image_init_output


class CG(nn.Module):
    def __init__(
        self,
        # forward_model,
        patch_size,
        nufft_im_size,
        ch_pad,
        iterations,
        gamma_init=5e-2,
        tau_init=1.0,
    ):
        super().__init__()
        self.forward_model = MR_Forward_Model_Static(patch_size, nufft_im_size)
        self.regularization = Regularization(ch_pad)
        # self.regularization = Identity_Regularization()
        self.iterations = iterations
        self.gamma = torch.nn.Parameter(
            torch.ones(self.iterations, dtype=torch.float32) * gamma_init
        )
        self.tau = torch.nn.Parameter(
            torch.ones(self.iterations, dtype=torch.float32) * tau_init
        )
        # self.gamma = 1e-1*torch.ones(self.iterations, dtype=torch.float32)
        # self.tau = 3*torch.ones(self.iterations, dtype=torch.float32)
        self.downsample = lambda x: interpolate(
            x, scale_factor=(1, 0.5, 0.5), mode="trilinear"
        )
        self.nufft_adj = tkbn.KbNufftAdjoint(im_size=nufft_im_size)

    def objective_function(self, params, kspace_data, kspace_traj):
        kspace_data_estimated = self.forward_model(params)
        return 1 / 2 * torch.norm(kspace_data_estimated - kspace_data, 1)

    def forward(self, kspace_data, kspace_traj, weights):
        # print(self.gamma)
        # print(self.tau)
        # initialization
        image_multi_ch = self.nufft_adjoint(kspace_data, kspace_traj)
        self.ph_num = image_multi_ch.shape[1]
        image_init = self.image_init(image_multi_ch)
        csm_init = self.csm_kernel_init(image_multi_ch)
        mvf_init = self.mvf_kernel_init(image_init)
        params = {
            "image": image_init.requires_grad_(True),
            "csm": csm_init.requires_grad_(True),
            "mvf": mvf_init.requires_grad_(True),
        }

        for t in range(self.iterations):
            # apply forward model to get kspace_data_estimated
            self.forward_model.generate_forward_operators(
                params["mvf"], params["csm"], kspace_traj
            )
            kspace_data_estimated = self.forward_model(params)
            loss_dc = 1 / 2 * torch.norm(kspace_data_estimated - kspace_data, 1)
            # grad_dc = torch.autograd.grad(loss_dc, params["csm"],)
            grad_dc = torch.autograd.grad(
                loss_dc, (params["image"], params["csm"], params["mvf"])
            )
            grad_dc = {
                "image": grad_dc[0],
                "csm": grad_dc[1],
                "mvf": grad_dc[2],
            }
            # loss_reg = self.regularization(params)
            # grad_reg = pytree.tree_map(lambda x, reg: x - reg, params, loss_reg)
            updates = self.update(grad_dc, t)
            params = self.apply_updates(params, updates)

            print(f"t: {t}, loss: {loss_dc}")
            # breakpoint()
        print(f"gamma: {self.gamma}, tau: {self.tau}")
        return params, image_init

    def update(self, dc_grads, reg_grads, t):
        return pytree.tree_map(
            lambda dc_grad, reg_grad: self.gamma[t]
            * (dc_grad + self.tau[t] * reg_grad),
            dc_grads,
            reg_grads,
        )

    def apply_updates(self, params, updates):
        return pytree.tree_map(lambda param, update: param - update, params, updates)

    def image_init(self, image_multi_ch):
        image_init = torch.sum(image_multi_ch * image_multi_ch.conj(), dim=2)[:, 0:1]
        return image_init

    def csm_kernel_init(self, image_multi_ch):
        b, ph, _, _, _, _ = image_multi_ch.shape
        csm_kernel = rearrange(image_multi_ch, "b ph ch d h w -> (b ph) ch d h w")
        for i in range(3):
            csm_kernel = self.downsample(csm_kernel)
        return rearrange(csm_kernel, "(b ph) ch d h w -> b ph ch d h w", b=b, ph=ph)

    def mvf_kernel_init(self, image):
        b, _, d, h, w = image.shape
        mvf_kernels = [
            torch.zeros((b, 3, d, h // 2, w // 2), device=image.device)
            for i in range(self.ph_num - 1)
        ]
        return torch.stack(mvf_kernels, dim=1)

    def nufft_adjoint(self, kspace_data, kspace_traj):
        b, ph, ch, d, length = kspace_data.shape
        images = []
        for k, kj in zip(
            torch.unbind(kspace_data, dim=1), torch.unbind(kspace_traj, dim=1)
        ):
            image = self.nufft_adj(
                rearrange(k, "b ch d len -> b (ch d) len"), kj, norm="ortho"
            )
            images.append(
                rearrange(image, "b (ch d) h w -> b ch d h w", b=b, ch=ch, d=d)
            )
        return torch.stack(images, dim=1)
