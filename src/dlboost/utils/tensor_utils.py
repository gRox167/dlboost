import torch 
import einops as eo

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
