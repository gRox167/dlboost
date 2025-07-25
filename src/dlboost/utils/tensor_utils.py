import re
from functools import partial
from types import NoneType
from typing import (
    Dict,
    List,
    Pattern,
    Sequence,
    Tuple,
    Union,
)

import einx
import torch
from jaxtyping import PyTree, Shaped
from mrboost.type_utils import ComplexImage3D
from optree import tree_map, tree_structure, tree_transpose
from plum import dispatch, overload
from sklearn.model_selection import KFold
from torch import Tensor
from torch.nn import functional as F
from xarray import DataArray


class GridSample3dForward(torch.autograd.Function):
    @staticmethod
    def forward(input, grid, align_corners):
        assert input.ndim == 5 and grid.ndim == 5
        output = F.grid_sample(
            input=input,
            grid=grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=align_corners,
        )
        return output

    # NEW: only runs when higher-order AD needs the saved tensors
    @staticmethod
    def setup_context(ctx, inputs, output):
        input, grid, align_corners = inputs  # tuple unpack
        ctx.save_for_backward(input, grid)  # what backward() needs
        ctx.align_corners = align_corners

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        align_corners = ctx.align_corners
        grad_input, grad_grid = GridSample3dBackward.apply(
            grad_output, input, grid, align_corners
        )
        return grad_input, grad_grid, None


class GridSample3dBackward(torch.autograd.Function):
    @staticmethod
    def forward(grad_output, input, grid, align_corners):
        op = torch.ops.aten.grid_sampler_3d_backward  # raw ATen kernel
        grad_input, grad_grid = op(
            grad_output,
            input,
            grid,
            0,  # mode  (0 = bilinear/trilinear)
            0,  # padding_mode (0 = zeros)
            align_corners,  # align_corners (False in forward)
            (True, True),  # output_mask      (return both grads)
        )
        return grad_input, grad_grid

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, _, grid, align_corners = inputs  # we only need 'grid' later
        ctx.save_for_backward(grid)
        ctx.align_corners = align_corners

    @staticmethod
    def backward(ctx, grad2_grad_input, grad2_grad_grid):
        _ = grad2_grad_grid  # unused
        (grid,) = ctx.saved_tensors
        align_corners = ctx.align_corners

        # 2nd-order gradient w.r.t. grad_output
        grad2_grad_output = (
            GridSample3dForward.apply(grad2_grad_input, grid, align_corners)
            if ctx.needs_input_grad[0]
            else None
        )

        # we stop gradients here for 'input' and 'grid'
        return grad2_grad_output, None, None, None


def complex_as_real_2ch(x):
    return einx.rearrange("b ch ... cmplx-> b (ch cmplx) ...", torch.view_as_real(x))


def tensor_memory(tensor: torch.Tensor, verbose: bool = True, unit="GB") -> float:
    """
    Returns the memory usage of a tensor in megabytes (MB).
    """
    if verbose:
        print(
            f"Tensor shape: {tensor.shape}, "
            f"element size: {tensor.element_size()} bytes, "
            f"number of elements: {tensor.nelement()}"
        )
    if unit == "GB":
        size = tensor.element_size() * tensor.nelement() / (1024**3)
    elif unit == "MB":
        size = tensor.element_size() * tensor.nelement() / (1024**2)
    elif unit == "KB":
        size = tensor.element_size() * tensor.nelement() / (1024**1)
    if verbose:
        print(f"Tensor memory usage: {size:.2f} {unit}")
    return size


def real_2ch_as_complex(x, ch=1):
    return torch.view_as_complex(
        einx.rearrange(
            "b (ch cmplx) ... -> b ch ... cmplx", x, ch=ch, cmplx=2
        ).contiguous()
    )


def complex_as_real_ch(func, ch=1):
    def wrapper(*args, **kwargs):
        _args = tree_map(complex_as_real_2ch, args)
        x = func(*_args, **kwargs)
        x = real_2ch_as_complex(x, ch=ch)
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


def percentile(t: torch.tensor, l, h):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    l_ = 1 + round(0.01 * float(l) * (t.numel() - 1))
    h_ = 1 + round(0.01 * float(h) * (t.numel() - 1))
    l_th = t.kthvalue(l_).values
    h_th = t.kthvalue(h_).values
    return l_th, h_th


def complex_normalize_abs_95(x, start_dim=0, expand=True):
    """
    Normalize the input complex tensor by clamping its absolute values
    between the 2.5th and 97.5th percentiles, then standardizing it.

    Args:
        x (torch.Tensor): Input complex tensor.
        start_dim (int, optional): The dimension to start flattening. Defaults to 0.
        expand (bool, optional): Whether to expand the mean and std to match the shape of x. Defaults to True.

    Returns:
        tuple: A tuple containing the normalized tensor, mean, and std.
    """
    x_abs = x.abs()
    min_95, max_95 = percentile(x_abs.flatten(start_dim), 2.5, 97.5)
    x_abs_clamped = torch.clamp(x_abs, min_95, max_95)
    mean = torch.mean(x_abs_clamped)
    std = torch.std(x_abs_clamped, unbiased=False)
    return (
        # (x - mean) / std,
        mean.expand_as(x_abs_clamped) if expand else mean,
        std.expand_as(x_abs_clamped) if expand else std,
    )


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


@overload
def interpolate(
    img: Shaped[ComplexImage3D, "b ch"],
    scale_factor: Sequence = (1.0, 1.0, 1.0),
    mode="trilinear",
    align_corners=True,
):
    if not torch.is_complex(img):
        return F.interpolate(
            img,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    else:
        r = F.interpolate(
            img.real,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
        i = F.interpolate(
            img.imag,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
        return torch.complex(r, i)


@overload
def interpolate(
    img: ComplexImage3D,
    scale_factor: Sequence = (1.0, 1.0, 1.0),
    mode="trilinear",
    align_corners=True,
):
    return (
        interpolate(
            img.unsqueeze(0).unsqueeze(0),
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
        .squeeze(0)
        .squeeze(0)
    )


@overload
def interpolate(
    img: Shaped[ComplexImage3D, "b"],
    scale_factor: Sequence = (1.0, 1.0, 1.0),
    mode="trilinear",
    align_corners=True,
):
    return interpolate(
        img.unsqueeze(1),
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
    ).squeeze(1)


@overload
def interpolate(
    img: Shaped[ComplexImage3D, "*b0 b1 b2 ch"],
    scale_factor: Sequence = (1.0, 1.0, 1.0),
    mode="trilinear",
    align_corners=True,
):
    b, *ph, ch, d, h, w = img.shape
    img = interpolate(
        einx.rearrange("b ph... ch d h w -> (b ph...) ch d h w", img),
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
    )
    img = einx.rearrange("(b ph...) ch d h w -> b ph... ch d h w", img, ph=ph)
    return img


@dispatch
def interpolate(
    img,
    scale_factor,
    mode,
    align_corners,
):
    pass


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
    return F.pad(data, pad_config, mode=mode, value=value)


def pad(data: DataArray, pad_sizes, mode="constant", value=0):
    # Apply padding using xarray's pad function
    # common_keys = data.dims & pad_sizes.keys()
    # _pad_sizes = {k: pad_sizes[k] for k in common_keys}
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
    return tree_map(lambda x, y: x.isel(y), data, slices)


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


def hybrid_kfold_split(
    items: List[str],
    fixed_pattern: Union[str, Pattern],
    n_splits: int = 5,
    fold_idx: int = 0,
    random_state: int = 42,
    return_fix_idx=False,
    verbose: bool = False,
) -> Tuple[List[int], List[int]]:
    """
    Performs a hybrid K-fold split where some items are fixed in training set based on a pattern,
    while others participate in cross-validation.

    Args:
        items: List of items (e.g., file paths, IDs) to split
        fixed_pattern: Regex pattern to identify items that should always be in training set
        n_splits: Number of folds for cross-validation
        fold_idx: Which fold to use (0 to n_splits-1)
        random_state: Random seed for reproducibility
        verbose: Whether to print debug information

    Returns:
        Tuple of (train_indices, val_indices)
    """
    # Find items matching the fixed pattern
    if isinstance(fixed_pattern, str):
        fixed_pattern = re.compile(fixed_pattern)

    m_list = [fixed_pattern.search(item) for item in items]

    # Items that should always be in training set
    fixed_train_idx = [i for i, m in enumerate(m_list) if m is not None]

    # Items that participate in cross-validation
    cv_idx = [i for i, m in enumerate(m_list) if m is None]

    # Perform k-fold split on the cross-validation items
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_splits = list(kf.split(cv_idx))
    cv_train_idx, cv_val_idx = cv_splits[fold_idx]

    # Combine fixed training indices with cross-validation training indices
    train_idx = fixed_train_idx + [cv_idx[i] for i in cv_train_idx]
    val_idx = [cv_idx[i] for i in cv_val_idx]

    if verbose:
        ic(train_idx, val_idx)
    if return_fix_idx:
        return train_idx, val_idx, fixed_train_idx
    return train_idx, val_idx


def collate_fn(batch):
    # transpose dict structure out of list
    batch_transposed = tree_transpose(
        tree_structure([0 for _ in batch], none_is_leaf=True),
        tree_structure(batch[0], none_is_leaf=True),
        batch,
    )
    return {
        k: torch.stack(v) if torch.is_tensor(v[0]) else v
        for k, v in batch_transposed.items()
    }


# class GridSample2dForward(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, grid):
#         assert input.ndim == 4
#         assert grid.ndim == 4
#         output = torch.nn.functional.grid_sample(
#             input=input,
#             grid=grid,
#             mode="bilinear",
#             padding_mode="zeros",
#             align_corners=False,
#         )
#         ctx.save_for_backward(input, grid)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, grid = ctx.saved_tensors
#         grad_input, grad_grid = GridSample2dBackward.apply(grad_output, input, grid)
#         return grad_input, grad_grid


# class GridSample2dBackward(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, grad_output, input, grid):
#         op = torch._C._jit_get_operation("aten::grid_sampler_2d_backward")
#         grad_input, grad_grid = op(grad_output, input, grid, 0, 0, False)
#         ctx.save_for_backward(grid)
#         return grad_input, grad_grid

#     @staticmethod
#     def backward(ctx, grad2_grad_input, grad2_grad_grid):
#         _ = grad2_grad_grid  # unused
#         (grid,) = ctx.saved_tensors
#         grad2_grad_output = None
#         grad2_input = None
#         grad2_grid = None

#         if ctx.needs_input_grad[0]:
#             grad2_grad_output = GridSample2dForward.apply(grad2_grad_input, grid)

#         assert not ctx.needs_input_grad[2]
#         return grad2_grad_output, grad2_input, grad2_grid
