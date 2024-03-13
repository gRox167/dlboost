from functools import partial
from types import NoneType
from typing import (
    Dict,
    Sequence,
    Union,
)

import einops as eo
import torch
from jaxtyping import PyTree
from optree import tree_map
from plum import dispatch, overload
from torch import Tensor
from torch.nn import functional as f
from xarray import DataArray

from dlboost.utils.type_utils import Data_With_Location, XArrayDevice


@overload
def _transfer_to_device(data: torch.Tensor, device: torch.device) -> torch.Tensor:
    return data.to(device)


@overload
def _transfer_to_device(
    data: Data_With_Location, device: torch.device
) -> Data_With_Location:
    return (_transfer_to_device(data[0], device), data[1])


@overload
def _transfer_to_device(
    data: Dict[str, Data_With_Location | torch.Tensor], device: torch.device
) -> Dict[str, Data_With_Location | torch.Tensor]:
    return {k: _transfer_to_device(v, device) for k, v in data.items()}


@overload
def _transfer_to_device(
    data: PyTree[DataArray, "T"], device: torch.device
) -> PyTree[torch.Tensor, "T"]:
    return tree_map(lambda x: torch.from_numpy(x.values).to(device), data)

# @overload
# def _transfer_to_device(
#     data: torch.Tensor,
#     device: XArrayDevice,
#     dims: Sequence[str],
# ) -> PyTree[DataArray, "T"]:
#     return DataArray( data= data.cpu().numpy(),dims = dims)


@overload
def _transfer_to_device(
    data: PyTree[torch.Tensor, "T"],
    device: XArrayDevice,
    dims: PyTree[Sequence[str], "T"],
) -> PyTree[DataArray, "T"]:
    return tree_map(lambda x, d: DataArray( data= x.cpu().numpy(),dims = d), data, dims)


@dispatch
def _transfer_to_device(data, device):
    pass


def complex_as_real_2ch(x):
    # breakpoint()
    if len(x.shape) == 4:
        return eo.rearrange(torch.view_as_real(x), "b c h w cmplx-> b (c cmplx) h w")
    elif len(x.shape) == 5:
        return eo.rearrange(
            torch.view_as_real(x), "b c d h w cmplx-> b (c cmplx) d h w"
        )


def real_2ch_as_complex(x, c=1):
    if len(x.shape) == 4:
        return torch.view_as_complex(
            eo.rearrange(
                x, "b (c cmplx) h w -> b c h w cmplx", c=c, cmplx=2
            ).contiguous()
        )
    elif len(x.shape) == 5:
        return torch.view_as_complex(
            eo.rearrange(
                x, "b (c cmplx) d h w -> b c d h w cmplx", c=c, cmplx=2
            ).contiguous()
        )


def complex_as_real_ch(func):
    def wrapper(x):
        x = complex_as_real_2ch(x)
        x = func(x)
        x = real_2ch_as_complex(x)
        return x

    return wrapper


def abs_real_2ch(x):
    return torch.complex(x[..., 0], x[..., 1]).abs()


def ifft2(x):
    # x_ = torch.fft.ifftshift(x, dim = (-2,-1))
    x_ = torch.fft.ifft2(x, norm="ortho")
    return x_


def fft2(x):
    x_ = torch.fft.fft2(x, norm="ortho")
    # x_ = torch.fft.fftshift(x_, dim = (-2,-1))
    return x_


def normalize(x, return_mean_std=False):
    mean = x.mean()
    std = x.std()
    if return_mean_std:
        return (x - mean) / std, mean, std
    else:
        return (x - mean) / std


def renormalize(x, mean, std):
    return x * std + mean


def formap(func, in_dims=0, out_dims=0, batch_size=1):
    def func_return(*args, **kwargs):
        if isinstance(in_dims, int):
            _in_dims = [in_dims] * len(args)
        elif isinstance(in_dims, Sequence):
            assert len(in_dims) == len(args)
            _in_dims = in_dims
        b = args[0].shape[_in_dims[0]]
        # for i, arg in enumerate(args):
        #     print(i, arg.shape)
        _args = [
            [arg] * b
            if i is None
            else torch.chunk(arg, dim=i, chunks=arg.shape[i] // batch_size + 1)
            for i, arg in zip(_in_dims, args)
        ]
        # for i, arg in enumerate(_args):
        #     print(i, len(arg))
        #     print(arg[0].shape)
        # _kwargs = {k: [v]*b for k, v in kwargs.items()}
        func_partial = partial(func, **kwargs)
        # _out = list(zip(*map(func_partial, *_args)))
        if isinstance(out_dims, int):
            _out = list(map(func_partial, *_args))
            _out = torch.cat(_out, dim=out_dims)
        elif isinstance(out_dims, Sequence):
            _out_dims = out_dims
            _out = list(zip(*map(func_partial, *_args)))
            _out = [
                torch.cat(out, dim=i) if i is not None else out
                for i, out in zip(_out_dims, _out)
            ]
        return _out if len(_out) > 1 else _out[0]

    return func_return


def for_vmap(func, in_dims=0, out_dims=0, batch_size: Union[int, None] = None):
    def func_return(*args, **kwargs):
        if isinstance(in_dims, int):
            _in_dims = [in_dims] * len(args)
        elif isinstance(in_dims, Sequence):
            assert len(in_dims) == len(args)
            _in_dims = in_dims
        b = args[0].shape[_in_dims[0]]
        # for i, arg in enumerate(args):
        #     print(i, arg.shape)
        if batch_size is None:
            _args = [
                [arg] * b if i is None else torch.unbind(arg, dim=i)
                for i, arg in zip(_in_dims, args)
            ]
        else:
            _args = [
                [arg] * b if i is None else torch.split(arg, batch_size, dim=i)
                for i, arg in zip(_in_dims, args)
            ]
        # for i, arg in enumerate(_args):
        #     print(i, len(arg))
        #     print(arg[0].shape)
        # _kwargs = {k: [v]*b for k, v in kwargs.items()}
        func_partial = partial(func, **kwargs)
        # _out = list(zip(*map(func_partial, *_args)))
        combine_func = torch.stack if batch_size is None else torch.cat
        if isinstance(out_dims, int):
            _out = list(map(func_partial, *_args))
            _out = combine_func(_out, dim=out_dims)
        elif isinstance(out_dims, Sequence):
            _out_dims = out_dims
            _out = list(zip(*map(func_partial, *_args)))
            _out = [
                combine_func(out, dim=i) if i is not None else out
                for i, out in zip(_out_dims, _out)
            ]
        elif isinstance(out_dims, NoneType):
            _out = list(map(func_partial, *_args))
            _out = None
        return _out

    return func_return


# def for_map(func, in_dims = 0, out_dims = 0):
#     def func_return(*args, **kwargs):
#         if isinstance(in_dims, int):
#             _in_dims = [in_dims] * len(args)
#         elif isinstance(in_dims, Sequence):
#             assert len(in_dims) == len(args)
#             _in_dims = in_dims
#         b = args[0].shape[_in_dims[0]]
#         # for i, arg in enumerate(args):
#         #     print(i, arg.shape)
#         _args = [ [arg]*b if i is None else torch.unbind(arg, dim = i) for i, arg in zip(_in_dims, args) ]
#         # for i, arg in enumerate(_args):
#         #     print(i, len(arg))
#         #     print(arg[0].shape)
#         # _kwargs = {k: [v]*b for k, v in kwargs.items()}
#         func_partial = partial(func, **kwargs)
#         # _out = list(zip(*map(func_partial, *_args)))
#         if isinstance(out_dims, int):
#             _out = list(map(func_partial, *_args))
#             _out = torch.stack(_out, dim = out_dims)
#         elif isinstance(out_dims, Sequence):
#             _out_dims = out_dims
#             _out = list(zip(*map(func_partial, *_args)))
#             _out = [torch.stack(out, dim = i) if i is not None else out for i, out in zip(_out_dims, _out)]
#         return _out if len(_out) > 1 else _out[0]


def interpolate(img, scale_factor, mode, align_corners=True):
    if not torch.is_complex(img):
        return f.interpolate(
            img, scale_factor=scale_factor, mode=mode, align_corners=True
        )
    else:
        r = f.interpolate(
            img.real, scale_factor=scale_factor, mode=mode, align_corners=True
        )
        i = f.interpolate(
            img.imag, scale_factor=scale_factor, mode=mode, align_corners=True
        )
        return torch.complex(r, i)


def pad(data: Tensor, dims, pad_sizes, mode="constant", value=0):
    # Create a padding configuration tuple
    # The tuple should have an even number of elements, with the form:
    # (pad_left, pad_right, pad_top, pad_bottom, ...)
    # Since we want to pad a specific dimension 'dim', we need to calculate
    # the correct positions in the tuple for pad_size_1 and pad_size_2.

    # Initialize padding configuration with zeros for all dimensions
    pad_config = [0] * (data.dim() * 2)
    # Set the padding sizes for the specified dimensions
    for dim, (pad_size_1, pad_size_2) in zip(dims, pad_sizes):
        # Calculate the indices in the padding configuration tuple
        # PyTorch pads from the last dimension backwards, so we need to invert the dimension index
        pad_index_1 = -(dim + 1) * 2
        pad_index_2 = pad_index_1 + 1

        # Set the desired padding sizes at the correct indices
        pad_config[pad_index_1] = pad_size_1
        pad_config[pad_index_2] = pad_size_2

    # Convert the list to a tuple and apply padding
    pad_config = tuple(pad_config)
    return f.pad(data, pad_config, mode=mode, value=value)


def pad(data: DataArray, pad_sizes, mode="constant", value=0):
    # Apply padding using xarray's pad function
    padded_DataArray = data.pad(pad_sizes, mode=mode, constant_values=value)
    return padded_DataArray


@overload
def crop(data: Tensor, dims, start_indices, crop_sizes) -> Tensor:
    # Initialize slicing configuration with colons for all dimensions
    slice_config = [slice(None)] * data.dim()

    # Set the slicing configuration for the specified dimensions
    for dim, start_index, crop_size in zip(dims, start_indices, crop_sizes):
        # crop_size = crop_sizes[dim]
        slice_config[dim] = slice(start_index, start_index + crop_size)

    # Apply slicing
    return data[tuple(slice_config)]


# @overload
# def crop(
#     data: DataArray, start_indices: Dict[str, int], crop_sizes: Dict[str, int]
# ) -> DataArray:
#     # Apply cropping using xarray's isel function
#     slices = {k: slice(start_indices[k], start_indices[k] + crop_sizes[k]) for k in start_indices.keys()}
#     return data.isel(slices)

@overload
def crop(
        data: PyTree[DataArray, "T"],
        start_indices: PyTree[Dict[str, int], "T"],
        crop_sizes: PyTree[Dict[str, int], "T"],
) -> PyTree[DataArray, "T"]:
    slices = tree_map(lambda x, y: slice(x, x + y), start_indices, crop_sizes)
    return tree_map(
        lambda x, y: x.isel(y), data, slices
    )

# @overload 
# def crop(
#     data: PyTree[torch.Tensor, "T"],
#     start_indices: PyTree[Dict[str, int], "T"]
#     crop_sizes: PyTree[Dict[str, int], "T"],
# ) -> PyTree[torch.Tensor, "T"]:
#     return tree_map(
#         lambda x, y, z: crop(x, y, z), data, start_indices, crop_sizes
#     )


@dispatch
def crop(data, dims, start_indices, crop_sizes):
    pass