from meerkat import image
import torch
import zarr
import torchkbnufft as tkbn
from einops import rearrange, reduce, repeat
from monai.inferers import sliding_window_inference
from dlboost.utils import to_png

def predict_step(batch, nufft_adj,predictor=None, patch_size=(1,320, 320), device=torch.device("cuda"), ch_reduce_fn = torch.sum):
    b, t, ch, ph, z, sp = batch["kspace_data"].shape
    # kspace_traj torch.Size([1, 2, 1, 5, 2, 9600])
    kspace_traj = batch["kspace_traj"].expand(
        -1, -1, ch, -1, -1, -1)
    #  cse torch.Size([b, ch, z, 320, 320])
    cse = rearrange(
        batch["cse"], "b ch z x y -> b () ch () z x y").expand(-1, t, -1, ph, -1, -1, -1)
    # cse torch.Size([b, t, ch, ph, z, 320, 320])
    image_recon, image_init = multi_contrast_predict_v(
        batch["kspace_data_compensated"],
        kspace_traj, cse,
        nufft_adj, predictor=predictor, patch_size=patch_size,
        sw_device=device,
        ch_reduce_fn=ch_reduce_fn)
    return image_recon.abs(), image_init.abs()#.mT.flip(-2)

def validation_step(self, batch, batch_idx, predictor=None, ch_reduce_fn = torch.sum):
    predictor = self if predictor is None else predictor
    b, t, ch, ph, z, sp = batch["kspace_data"].shape
    # kspace_traj torch.Size([1, 2, 1, 5, 2, 9600])
    batch["kspace_traj"] = batch["kspace_traj"].expand(
        -1, -1, ch, -1, -1, -1)
    # cse torch.Size([b, ch, z, 320, 320])
    batch["cse"] = rearrange(
        batch["cse"], "b ch z x y -> b () ch () z x y").expand(-1, t, -1, ph, -1, -1, -1)
    # cse torch.Size([b, t, ch, ph, z, 320, 320])
    image_recon, image_init = multi_contrast_predict_v(
        batch["kspace_data_compensated"][:, :, :, :, 40:50],
        batch["kspace_traj"], batch["cse"][:, :, :, :, 40:50],
        self.nufft_adj, predictor=predictor, patch_size=self.patch_size,
        sw_device=self.device, sum_reduce_fn=torch.sum)
    # print(image_recon.shape, image_init.shape)
    zarr.save(self.trainer.default_root_dir +
              f'/epoch_{self.trainer.current_epoch}_recon.zarr',
              image_recon[0, 0].abs().mT.flip(-2).numpy(force=True))
    zarr.save(self.trainer.default_root_dir +
              f'/epoch_{self.trainer.current_epoch}_init.zarr',
              image_init[0, 0].abs().mT.flip(-2).numpy(force=True))
    to_png(self.trainer.default_root_dir +
           f'/epoch_{self.trainer.current_epoch}_image_recon.png',
           image_recon[0, 0, 0, 0].mT.flip(-2))
    print("saved to "+f'/epoch_{self.trainer.current_epoch}_image_recon.png')
    to_png(self.trainer.default_root_dir +
           f'/epoch_{self.trainer.current_epoch}_image_init.png',
           image_init[0, 0, 0, 0].mT.flip(-2))


def multi_contrast_predict_v(kspace_data_compensated, kspace_traj, cse, nufft_adj_op, predictor, patch_size, sw_device, ch_reduce_fn = torch.sum):
    r = [forward_contrast(kd, kt, c, nufft_adj_op, predictor, patch_size, sw_device,ch_reduce_fn) for kd, kt, c in zip(
        kspace_data_compensated.unbind(0), kspace_traj.unbind(0), cse.unbind(0))]
    image_recon, image_init = zip(*r)
    image_recon = torch.stack(image_recon, dim=0)
    image_init = torch.stack(image_init, dim=0)
    return image_recon, image_init


def forward_contrast(kspace_data_compensated, kspace_traj, cse, nufft_adj_op, predictor, patch_size, sw_device, ch_reduce_fn = torch.sum):
    r = [forward_ch(kd, kt, c, nufft_adj_op, predictor, patch_size, sw_device, ch_reduce_fn) for kd, kt, c in zip(
        kspace_data_compensated.unbind(0), kspace_traj.unbind(0), cse.unbind(0))]
    image_recon, image_init = zip(*r)
    image_recon = torch.stack(image_recon, dim=0)
    image_init = torch.stack(image_init, dim=0)
    return image_recon, image_init


def forward_ch(kspace_data_compensated, kspace_traj, cse, nufft_adj_op, predictor, patch_size, sw_device, ch_reduce_fn = torch.sum):
    # kspace_traj_ch = kspace_traj[0] if kspace_traj.shape[0] == 1 else kspace_traj
    image_recon, image_init = forward_step(
        kspace_data_compensated=kspace_data_compensated, kspace_traj=kspace_traj, cse=cse, nufft_adj_op=nufft_adj_op, predictor=predictor, patch_size=patch_size, sw_device = sw_device)
    return ch_reduce_fn(image_recon, 0), ch_reduce_fn(image_init, 0)


def forward_step(kspace_data_compensated, kspace_traj, cse, nufft_adj_op, predictor, patch_size, sw_device):
    image_init = nufft_adj_fn(kspace_data_compensated,
                              kspace_traj, nufft_adj_op.to(torch.device("cpu")))
    image_recon = sliding_window_inference(
        image_init, roi_size=patch_size,
        sw_batch_size=1, overlap=0, predictor=predictor.to(sw_device), device=torch.device("cpu"), sw_device=sw_device)  # , mode='gaussian')
    # image_recon = image_init.clone()
    return image_recon * cse.cpu().conj(), image_init.cpu() * cse.cpu().conj()


def nufft_fn(image, omega, nufft_op, norm="ortho"):
    """do nufft on image

    Args:
        image (_type_): b ph z x y
        omega (_type_): b ph complex_2ch l
        nufft_op (_type_): tkbn operator
        norm (str, optional): Defaults to "ortho".
    """
    b, ph, c, l = omega.shape
    image_kx_ky_z = nufft_op(  # torch.squeeze(image, dim=1)
        rearrange(image, "b ph z x y -> (b ph) z x y"),
        rearrange(omega, "b ph c l -> (b ph) c l"), norm=norm)
    image_kx_ky_z = rearrange(
        image_kx_ky_z, "(b ph) z l -> b ph z l", b=b)
    # image_kx_ky_z.unsqueeze_(dim=1)
    return image_kx_ky_z


def nufft_adj_fn(kdata, omega, nufft_adj_op, norm="ortho"):
    """do adjoint nufft on kdata  

    Args:
        kdata (_type_): b ph z l
        omega (_type_): b ph complex_2ch l
        nufft_adj_op (_type_): tkbn operator
        norm (str, optional): Defaults to "ortho".

    Returns:
        _type_: _description_
    """
    b, ph, c, l = omega.shape
    image = nufft_adj_op(rearrange(kdata, "b ph z l -> (b ph) z l"),
                         rearrange(omega, "b ph c l -> (b ph) c l"), norm=norm)
    return rearrange(image, "(b ph) z x y -> b ph z x y", b=b, ph=ph)


def generate_disjoint_masks(length, sections, device):
    rand_perm = torch.split(torch.randperm(length), sections)
    masks = []
    for perm in rand_perm:
        mask_base = torch.tensor([False]*length, device=device)
        mask_base[perm] = True
        masks.append(mask_base)
    return masks


def generate_image_mask(img, idx, width=4):
    n, d, h, w = img.shape
    mask = rearrange(torch.zeros_like(img, dtype=torch.bool),
                     "n d (h h1) (w w1) -> n d h w (h1 w1)", h1=width, w1=width)
    mask[..., idx] = True

    # mask = torch.zeros(size=(n*d*h*w, ),
    #                    device=img.device)
    # idx_list = torch.arange(
    #     0, width**2, device=img.device)
    # rd_pair_idx = idx_list[torch.tensor(idx).repeat(n * d * h * w //width**2 )]
    # rd_pair_idx += torch.arange(start=0,
    #                             end=n * d * h * w ,
    #                             step=width**2,
    #                             # dtype=torch.int64,
    #                             device=img.device)

    # mask[rd_pair_idx] = 1
    # mask = mask.view(n, d, h, w)
    return rearrange(mask, "n d h w (h1 w1) -> n d (h h1) (w w1)", h1=width, w1=width)


def interpolate_mask_3x3_weighted_avg(tensor, mask, mask_inv, interpolation_kernel):
    n, d, h, w = tensor.shape
    kernel = torch.tensor(interpolation_kernel, device=tensor.device)[
        None, None, :, :, :]
    kernel = kernel / kernel.sum()

    if tensor.dtype == torch.float16:
        kernel = kernel.half()
    elif tensor.dtype == torch.complex64:
        kernel = kernel+1j*kernel
        # how to do interpolation for complex number?
    # breakpoint()
    filtered_tensor = torch.nn.functional.conv3d(
        tensor.view(n, 1, d, h, w), kernel, stride=1, padding=1)

    return filtered_tensor.view_as(tensor) * mask + tensor * mask_inv


class ImageMasker(object):
    def __init__(self, width=4):
        self.width = width
        self.interpolation_kernel = [
            [[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]],
            [[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]],
            [[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]],
        ]

    def mask(self, img, idx):
        # This function generates masked images given random masks
        img_ = rearrange(img, 'n c d h w -> (n c) d h w')
        mask = generate_image_mask(img_, idx, width=self.width)
        mask_inv = torch.logical_not(mask)
        masked_img = interpolate_mask_3x3_weighted_avg(
            img_, mask, mask_inv, self.interpolation_kernel)

        masked_img = rearrange(
            masked_img, '(n c) d h w -> n c d h w', n=img.shape[0])
        mask = rearrange(mask, '(n c) d h w -> n c d h w', n=img.shape[0])
        mask_inv = rearrange(
            mask_inv, '(n c) d h w -> n c d h w', n=img.shape[0])
        return masked_img, mask, mask_inv

    # def train(self, img):
    #     n, d, h, w = img.shape
    #     tensors = torch.zeros((n, self.width**2, c, h, w), device=img.device)
    #     masks = torch.zeros((n, self.width**2, 1, h, w), device=img.device)
    #     for i in range(self.width**2):
    #         x, mask = self.mask(img, i)
    #         tensors[:, i, ...] = x
    #         masks[:, i, ...] = mask
    #     tensors = tensors.view(-1, c, h, w)
    #     masks = masks.view(-1, 1, h, w)
    #     return tensors, masks
