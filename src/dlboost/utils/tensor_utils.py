from types import NoneType
from sympy import N
import torch 
import einops as eo
from functools import partial
from typing import Sequence, Union
from torch.nn import functional as f

def complex_as_real_2ch(x):
    # breakpoint()
    if len(x.shape) == 4:
        return eo.rearrange(torch.view_as_real(x), "b c h w cmplx-> b (c cmplx) h w")
    elif len(x.shape)==5:
        return eo.rearrange(torch.view_as_real(x), "b c d h w cmplx-> b (c cmplx) d h w")

def real_2ch_as_complex(x, c = 1):
    if len(x.shape) == 4:
        return torch.view_as_complex(eo.rearrange(x, "b (c cmplx) h w -> b c h w cmplx",c=c, cmplx=2).contiguous())
    elif len(x.shape)==5:
        return torch.view_as_complex(eo.rearrange(x, "b (c cmplx) d h w -> b c d h w cmplx",c=c, cmplx=2).contiguous())

def complex_as_real_ch(func):
    def wrapper(x):
        x = complex_as_real_2ch(x)
        x = func(x)
        x = real_2ch_as_complex(x)
        return x
    return wrapper

def abs_real_2ch(x):
    return torch.complex(x[...,0], x[...,1]).abs()

def ifft2(x):
    # x_ = torch.fft.ifftshift(x, dim = (-2,-1))
    x_ = torch.fft.ifft2(x, norm = "ortho")
    return x_

def fft2(x):
    x_ = torch.fft.fft2(x, norm = "ortho")
    # x_ = torch.fft.fftshift(x_, dim = (-2,-1))
    return x_


def normalize(x, return_mean_std=False):
    mean = x.mean()
    std = x.std()
    if return_mean_std:
        return (x-mean)/std, mean, std
    else:
        return (x-mean)/std


def renormalize(x, mean, std):
    return x*std+mean

def formap(func, in_dims = 0, out_dims = 0, batch_size = 1):
    def func_return(*args, **kwargs):
        if isinstance(in_dims, int):
            _in_dims = [in_dims] * len(args)
        elif isinstance(in_dims, Sequence):
            assert len(in_dims) == len(args)
            _in_dims = in_dims
        b = args[0].shape[_in_dims[0]]
        # for i, arg in enumerate(args):
        #     print(i, arg.shape)
        _args = [ [arg]*b if i is None else torch.chunk(arg, dim = i, chunks = arg.shape[i]//batch_size+1) for i, arg in zip(_in_dims, args) ]
        # for i, arg in enumerate(_args):
        #     print(i, len(arg))
        #     print(arg[0].shape)
        # _kwargs = {k: [v]*b for k, v in kwargs.items()}
        func_partial = partial(func, **kwargs)
        # _out = list(zip(*map(func_partial, *_args)))
        if isinstance(out_dims, int):
            _out = list(map(func_partial, *_args))
            _out = torch.cat(_out, dim = out_dims)
        elif isinstance(out_dims, Sequence):
            _out_dims = out_dims
            _out = list(zip(*map(func_partial, *_args)))
            _out = [torch.cat(out, dim = i) if i is not None else out for i, out in zip(_out_dims, _out)]
        return _out if len(_out) > 1 else _out[0]
    return func_return

def for_vmap(func, in_dims = 0, out_dims = 0, batch_size: Union[int, None] = None):
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
            _args = [ [arg]*b if i is None else torch.unbind(arg, dim = i) for i, arg in zip(_in_dims, args) ]
        else:
            _args = [ [arg]*b if i is None else torch.split(arg, batch_size,dim = i) for i, arg in zip(_in_dims, args) ]
        # for i, arg in enumerate(_args):
        #     print(i, len(arg))
        #     print(arg[0].shape)
        # _kwargs = {k: [v]*b for k, v in kwargs.items()}
        func_partial = partial(func, **kwargs)
        # _out = list(zip(*map(func_partial, *_args)))
        combine_func = torch.stack if batch_size is None else torch.cat
        if isinstance(out_dims, int):
            _out = list(map(func_partial, *_args))
            _out = combine_func(_out, dim = out_dims)
        elif isinstance(out_dims, Sequence):
            _out_dims = out_dims
            _out = list(zip(*map(func_partial, *_args)))
            _out = [combine_func(out, dim = i) if i is not None else out for i, out in zip(_out_dims, _out)]
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


def interpolate(img, scale_factor, mode, align_corners = True):
    if not torch.is_complex(img):
        return f.interpolate(img, scale_factor=scale_factor, mode=mode, align_corners=True)
    else:
        r = f.interpolate(img.real, scale_factor=scale_factor, mode=mode, align_corners=True)
        i = f.interpolate(img.imag, scale_factor=scale_factor, mode=mode, align_corners=True)
        return torch.complex(r,i)
