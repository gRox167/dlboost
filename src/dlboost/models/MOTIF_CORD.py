import einx
import torch
from torch import nn

from dlboost.models import ComplexUnet, DWUNet, SpatialTransformNetwork
from dlboost.NODEO.Utils import resize_deformation_field
from dlboost.utils.tensor_utils import interpolate
from mrboost.computation import generate_nufft_op


class CSM_FixPh(nn.Module):
    def __init__(self):
        super().__init__()

        # if upsample_times:
        def upsample(x):
            for i in range(3):
                # ic(x.shape)
                x = interpolate(x, scale_factor=(1, 2, 2), mode="trilinear")
            return x

        self.upsample = upsample
        # self.upsample = vmap(upsample)

    def generate_forward_operator(self, csm_kernels):
        # _csm = self.upsample(csm_kernels)
        # print(_csm.shape)
        # self._csm = _csm / torch.sqrt(
        #     torch.sum(torch.abs(_csm) ** 2, dim=1, keepdim=True)
        # )
        self._csm = csm_kernels

    def forward(self, image):
        # ic(image.shape, self._csm.shape)
        return image.unsqueeze(2) * self._csm


class MVF_Dyn(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.spatial_transform = SpatialTransformNetwork(size=size, mode="bilinear")
        # if upsample_times:
        #     def upsample(x):
        #         for i in range(upsample_times):
        #             x = interpolate(x, scale_factor=(1, 2, 2), mode="trilinear")
        #         return x
        # else:
        #     upsample = lambda x: x  # noqa: E731
        # self.upsample = upsample

    def generate_forward_operator(self, mvf_kernels):
        self.ph_to_move = mvf_kernels.shape[1]
        _mvf = einx.rearrange("b ph v d h w -> (b ph) v d h w", mvf_kernels.clone())
        self._mvf = resize_deformation_field(_mvf, (1, 2, 2))
        # self._mvf = self.upsample(_mvf)

    def forward(self, image):
        # image is a tensor with shape b, 1, d, h, w, 2
        # image_ref = image.clone()

        image_move = einx.rearrange(
            "b () d h w comp-> (b ph) comp d h w",
            torch.view_as_real(image),
            ph=self.ph_to_move,
        )
        # rearrange the image to (b, ph), 2, d, h, w
        image_4ph = self.spatial_transform(image_move, self._mvf)
        image_4ph = einx.rearrange(
            "(b ph) comp d h w -> b ph d h w comp", image_4ph, ph=self.ph_to_move
        )
        image_4ph = torch.complex(image_4ph[..., 0], image_4ph[..., 1])

        return torch.cat((image, image_4ph), dim=1)


class NUFFT(nn.Module):
    def __init__(self, nufft_im_size):
        super().__init__()
        self.nufft_im_size = nufft_im_size
        nufft_op, nufft_adj_op = generate_nufft_op(nufft_im_size)
        self.nufft_cse = torch.vmap(nufft_op)  # image: b ch d h w; ktraj: b 2 len
        self.nufft_adj_cse = torch.vmap(nufft_adj_op)  # kdata: b ch d len
        self.nufft = torch.vmap(torch.vmap(nufft_op))
        # image: b ph ch d h w; ktraj: b ph 2 len
        self.nufft_adj = torch.vmap(torch.vmap(nufft_adj_op))
        # kdata: b ph ch d len

    def generate_forward_operator(self, kspace_traj):
        self.kspace_traj = kspace_traj

    def adjoint(self, kspace_data):
        return self.nufft_adj(kspace_data, self.kspace_traj)

    def forward(self, image):
        return self.nufft(image, self.kspace_traj)


class MR_Forward_Model_Static(nn.Module):
    def __init__(
        self,
        image_size,
        nufft_im_size,
        CSM_module=CSM_FixPh,
        MVF_module=MVF_Dyn,
        NUFFT_module=NUFFT,
    ):
        super().__init__()
        self.M = MVF_module(image_size)
        self.S = CSM_module()
        self.N = NUFFT_module(nufft_im_size)

    def generate_forward_operators(self, mvf_kernels, csm_kernels, kspace_traj):
        self.M.generate_forward_operator(mvf_kernels) if self.M else None
        self.S.generate_forward_operator(csm_kernels)
        self.N.generate_forward_operator(kspace_traj)

    def forward(self, _image):
        # This is needed, because in torchkbnufft there are inplace operation on leaf tensor (requires_grad = True)
        image_5ph = self.M(_image) if self.M else _image.expand(-1, 5, -1, -1, -1)
        image_5ph_multi_ch = self.S(image_5ph)
        kspace_data_estimated = self.N(image_5ph_multi_ch.clone())
        return kspace_data_estimated


class Regularization(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_denoiser = ComplexUnet(
            1,
            1,
            spatial_dims=3,
            conv_net=DWUNet(
                in_channels=2,
                out_channels=2,
                features=(16, 32, 64, 128, 256),
                # features=(32, 64, 128, 256, 512),
                strides=((2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2)),
                kernel_sizes=(
                    (3, 3, 3),
                    (3, 3, 3),
                    (3, 3, 3),
                    (3, 3, 3),
                    (3, 3, 3),
                ),
            ),
        )

    def forward(self, params):
        params = params.clone()
        return self.image_denoiser(params)


class Identity_Regularization:
    def __init__(self):
        pass

    def __call__(self, params):
        return params


class MOTIF_CORD(nn.Module):
    def __init__(
        self,
        patch_size: tuple = (16, 320, 320),
        patch_effective_ratio=0.2,
        nufft_im_size: tuple = (320, 320),
        epsilon: float = 1e-2,
        iterations: int = 5,
        gamma_init=0.1,
        tau_init=0.2,
    ):
        super().__init__()
        self.forward_model = MR_Forward_Model_Static(patch_size, nufft_im_size)
        self.regularization = Regularization()
        self.epsilon = epsilon
        self.iterations = iterations
        self.gamma = gamma_init
        self.tau = tau_init
        self.downsample = lambda x: interpolate(
            x, scale_factor=(1, 0.5, 0.5), mode="trilinear"
        )
        self.effective_slice = slice(
            int((1 - patch_effective_ratio) * patch_size[0] / 2),
            int((1 + patch_effective_ratio) * patch_size[0] / 2),
        )
        # self.nufft_adj = tkbn.KbNufftAdjoint(im_size=nufft_im_size)

    def forward(
        self, kspace_data, kspace_traj, image_init, mvf, csm, weights_flag=True
    ):
        # initialization
        # print(image_init[0, 0, 0])
        # from monai.visualize import matshow3d
        # matshow3d(
        #     image_init[0, 0, 0:5].abs().cpu().numpy(), cmap="gray", vmin=0, vmax=5
        # )
        # plt.imshow(image_init[0, 0, 40])
        self.ph_num = kspace_data.shape[1]
        image_list = []
        if torch.is_complex(image_init):
            x = image_init
        else:
            x = torch.complex(image_init, torch.zeros_like(image_init))
        image_list.append(image_init.cpu())

        self.forward_model.generate_forward_operators(mvf, csm, kspace_traj)
        # dc_loss = self.inner_loss(x, kspace_data)
        # print(f"t: 0, loss: {dc_loss}")
        x.requires_grad_(True)

        # grad_dc_fn = grad(lambda img: self.inner_loss(img, kspace_data))

        for t in range(1, self.iterations + 1):
            # apply forward model to get kspace_data_estimated
            dc_loss = self.inner_loss(x.clone(), kspace_data, weights_flag)
            grad_dc = torch.autograd.grad(dc_loss, x)[0]
            grad_reg = torch.zeros_like(x, dtype=torch.complex64)
            grad_reg[:, :, self.effective_slice] = x[
                :, :, self.effective_slice
            ] - self.regularization(x[:, :, self.effective_slice])
            updates = -(self.gamma * grad_dc + self.tau * grad_reg)
            # updates = -self.gamma * grad_dc
            x = x.add(updates)
            image_list.append(x.clone().detach().cpu())
            print(f"t: {t}, loss: {dc_loss}")

        return x, image_list

    def image_init(self, image_multi_ch, csm):
        image_init = torch.sum(image_multi_ch * csm.conj(), dim=2)
        return image_init

    def inner_loss(self, x, kspace_data, weights_flag):
        kspace_data_estimated = self.forward_model(x)
        if weights_flag:
            kspace_data_estimated_detatched = (
                kspace_data_estimated[:, :, :, self.effective_slice].detach().abs()
            )
            norm_factor = kspace_data_estimated_detatched.max()
            weights = 1 / (kspace_data_estimated_detatched / norm_factor + self.epsilon)
        else:
            weights = 1
        diff = torch.view_as_real(
            weights * kspace_data_estimated[:, :, :, self.effective_slice]
        ) - torch.view_as_real(weights * kspace_data[:, :, :, self.effective_slice])
        loss_dc = torch.mean(diff**2)
        return loss_dc
