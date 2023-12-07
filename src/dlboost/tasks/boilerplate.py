# from meerkat import image
import torch
# import torch.nn.functional as f
from einops import rearrange# , reduce, repeat

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


# def interpolate(img, scale_factor, mode):
#     if not torch.is_complex(img):
#         return f.interpolate(img, scale_factor=scale_factor, mode=mode, align_corners=True)
#     else:
#         r = f.interpolate(img.real, scale_factor=scale_factor, mode=mode, align_corners=True)
#         i = f.interpolate(img.imag, scale_factor=scale_factor, mode=mode, align_corners=True)
#         return torch.complex(r,i)
    

    