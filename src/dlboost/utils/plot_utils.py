# import cv2
import imageio
import numpy as np
import torch
from matplotlib import pyplot as plt

# def to_tiff(x, path, is_normalized=True):
#     try:
#         x = np.squeeze(x)
#     except:
#         pass

#     try:
#         x = torch.squeeze(x).numpy()
#     except:
#         pass

#     print(x.shape, path)

#     if len(x.shape) == 3:
#         n_slice, n_x, n_y = x.shape

#         if is_normalized:
#             for i in range(n_slice):
#                 x[i] -= np.amin(x[i])
#                 x[i] /= np.amax(x[i])

#                 x[i] *= 255

#             x = x.astype(np.uint8)

#     else:
#         n_slice, n_x, n_y, n_c = x.shape

#         x = np.expand_dims(x, -1)
#         # x = post_processing(np.expand_dims(x, -1))
#         x = np.squeeze(x)

#         if is_normalized:
#             for i in range(n_slice):
#                 for j in range(n_c):
#                     x[i, :, :, j] -= np.amin(x[i, :, :, j])
#                     x[i, :, :, j] /= np.amax(x[i, :, :, j])

#                     x[i, :, :, j] *= 255

#             x = x.astype(np.uint8)

#     tiff.imwrite(path, x, imagej=True, ijmetadata={'Slice': n_slice})


def to_gif(path, input_array, is_segmentation=False):
    array = input_array.clone().detach().cpu().numpy()
    if isinstance(array, torch.Tensor):
        # array = array.clamp(min=0, max=1).detach().cpu().numpy()
        array = array / array.max() if array.max() > 0.3 else array
    else:
        array = array / array.max() if array.max() > 0.3 else array
    if len(array.shape) == 4:
        if array.shape[0] > 1:
            array = (
                255 * array.max(axis=0)
                if not is_segmentation
                else 255 * array.argmax(axis=0)
            )
            # array[array>0] = 255
        else:
            array = 255 * array.squeeze()
    else:
        array = 255 * array
    length = array.shape[-1]
    with imageio.get_writer(path, mode="I") as writer:
        for i in range(length):
            writer.append_data(array[:, :, i].transpose().astype(np.uint8))


# optimize(path)


def to_png(path, input_array, vmin=None, vmax=None):
    if input_array.dtype == torch.float32:
        plt.imsave(
            path,
            input_array.clone().detach().cpu().numpy(),
            vmin=vmin,
            vmax=vmax,
            cmap="gray",
        )
    elif input_array.dtype == torch.complex64:
        plt.imsave(
            path,
            input_array.clone().detach().cpu().abs().numpy(),
            vmin=vmin,
            vmax=vmax,
            cmap="gray",
        )
    # plt.imsave('tests/image_recon_fixed_cache_filled_wrap.png',image_recon_fixed_cache[0,40,:,:].abs().cpu())
