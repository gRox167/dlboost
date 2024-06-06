# import typing
from collections import ChainMap
from itertools import product

# type
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Mapping,
    NotRequired,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
import torch
from beartype.door import is_bearable
from icecream import ic
from jaxtyping import PyTree
from optree import _C, tree_flatten, tree_map, tree_map_, tree_structure
from plum import dispatch, overload
from torch import Size, Tensor
from xarray import DataArray

from dlboost.utils.tensor_utils import crop, pad

Location = Sequence[slice]
Data_With_Location = Tuple[Any, Location]
PadSizes = Sequence[Tuple[int, int]]
NamedSize = Dict[str, int]
Patches_With_PadSizes = TypedDict(
    "Patches_With_PadSizes",
    {
        "patches": Sequence[Data_With_Location],
        "pad_sizes": NotRequired[PadSizes],
        "padded_size": NotRequired[Size],
    },
)

XArrayDevice = Literal["xarray", "disk", "xa"]


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
    return tree_map(lambda x, d: DataArray(data=x.cpu().numpy(), dims=d), data, dims)


@dispatch
def _transfer_to_device(data, device):
    pass


def tree_transpose_map(
    func,
    tree,
    *rests,
    inner_treespec=None,
    is_leaf=None,
    none_is_leaf=False,
    namespace: str = "",
):  # PyTree[PyTree[U]]
    """Map a multi-input function over pytree args to produce a new pytree with transposed structure.

    See also :func:`tree_map`, :func:`tree_map_with_path`, and :func:`tree_transpose`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)}
    >>> tree_transpose_map(  # doctest: +IGNORE_WHITESPACE
    ...     lambda x: {'identity': x, 'double': 2 * x},
    ...     tree
    ... )
    {
        'identity': {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)},
        'double': {'b': (4, [6, 8]), 'a': 2, 'c': (10, 12)}
    }
    >>> tree_transpose_map(  # doctest: +IGNORE_WHITESPACE
    ...     lambda x: {'identity': x, 'double': (x, x)},
    ...     tree,
    ...     inner_treespec=tree_structure({'identity': 0, 'double': 0})
    ... )
    {
        'identity': {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)},
        'double': {'b': ((2, 2), [(3, 3), (4, 4)]), 'a': (1, 1), 'c': ((5, 5), (6, 6))}
    }

    Args:
        func (callable): A function that takes ``1 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        inner_treespec (PyTreeSpec, optional): The treespec object representing the inner structure
            of the result pytree. If not specified, the inner structure is inferred from the result
            of the function ``func`` on the first leaf. (default: :data:`None`)
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A new nested pytree with the same structure as ``inner_treespec`` but with the value at each
        leaf has the same structure as ``tree``. The subtree at each leaf is given by the result of
        function ``func(x, *xs)`` where ``x`` is the value at the corresponding leaf in ``tree`` and
        ``xs`` is the tuple of values at corresponding nodes in ``rests``.
    """
    leaves, outer_treespec = _C.flatten(tree, is_leaf, none_is_leaf, namespace)
    if outer_treespec.num_leaves == 0:
        raise ValueError(
            f"The outer structure must have at least one leaf. Got: {outer_treespec}."
        )
    flat_args = [leaves] + [outer_treespec.flatten_up_to(r) for r in rests]
    outputs = list(map(func, *flat_args))

    if inner_treespec is None:
        inner_treespec = tree_structure(
            outputs[0],
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
    if inner_treespec.num_leaves == 0:
        raise ValueError(
            f"The inner structure must have at least one leaf. Got: {inner_treespec}."
        )

    grouped = [inner_treespec.flatten_up_to(o) for o in outputs]
    transposed = zip(*grouped)
    subtrees = map(outer_treespec.unflatten, transposed)
    return inner_treespec.unflatten(subtrees)


@overload
def _pad_for_scaning_windows(
    input: DataArray,
    patch_size: Dict[str, int],
    overlap: Dict[str, float],
) -> Tuple[DataArray, PadSizes, Dict[str, int]]:
    pad_sizes = tree_map(lambda x, y: (round(x * y),) * 2, patch_size, overlap)

    output = pad(input, pad_sizes, mode="constant", value=0)
    return output, pad_sizes


@dispatch
def _pad_for_scaning_windows(
    input: Any,
    patch_dims: Any,
    patch_size: Any,
    overlap: Any,
):
    pass


@overload
def _forward_for_scanning_windows(
    input_padded: PyTree[DataArray],
    patch_location: Sequence[Dict[str, Location]],
    func: Callable,
    output: PyTree[DataArray],
    filter_func: Callable,
    device: torch.device,
    merge_device: torch.device | XArrayDevice,
    *args: Any,
    **kwargs: Any,
):
    for loc in patch_location:
        data = tree_map(lambda x: x.isel(loc, missing_dims="ignore"), input_padded)
        data_device = _transfer_to_device(data, device)
        result: PyTree[torch.Tensor] = func(data_device, *args, **kwargs)
        output_dims = tree_map(lambda x: x.dims, output)
        result_merge_device = _transfer_to_device(result, merge_device, output_dims)
        output_isel = tree_map(lambda x: x.isel(loc, missing_dims="ignore"), output)
        tree_map_(lambda x, y: x + filter_func(y), output_isel, result_merge_device)
    return output


@dispatch
def _forward_for_scanning_windows(
    input_splitted,
    func,
    output,
    filter_func,
    key_component,
    device,
    storage_device,
    *args,
    **kwargs,
):
    pass


@overload
def generate_patch_location(
    patch_size: int,
    dimension_size: int,
    overlap: float,
) -> Sequence[slice]:
    if overlap == 0:
        return tuple(slice(i, i + 1) for i in range(0, dimension_size))
    else:
        step = int(patch_size * (1 - 2 * overlap))
        return tuple(
            slice(i, i + patch_size) for i in range(0, dimension_size - step, step)
        )


@overload
def generate_patch_location(
    patch_size: Dict[str, int],
    dimension_size: Dict[str, int],
    overlap: Dict[str, float],
) -> Tuple[Dict[str, Location]]:
    indices: Dict[str, Sequence[slice]] = tree_map(
        lambda p, o, d: generate_patch_location(p, o, d),
        patch_size,
        dimension_size,
        overlap,
    )
    keys, values = zip(*indices.items())
    return tuple(dict(zip(keys, v)) for v in product(*values))


@dispatch
def generate_patch_location(patch_size, overlap, dimension_size):
    pass


def _is_named_size(x):
    if flag := is_bearable(x, dict):
        for k, v in x.items():
            flag = flag and isinstance(k, str) and isinstance(v, int)
        return flag
    else:
        return flag


def _is_pad_size(x):
    if flag := is_bearable(x, dict):
        for k, v in x.items():
            flag = flag and isinstance(k, str) and isinstance(v, tuple) and len(v) == 2
        return flag
    else:
        return flag


def merge_dicts(dicts: PyTree[Dict], is_leaf=_is_named_size) -> Dict:
    size_dicts = tree_flatten(dicts, is_leaf=is_leaf)[0]
    return dict(ChainMap(*size_dicts))


def get_dim_size_dict_from_tree(
    input: PyTree[DataArray], dim_names: Sequence[str]
) -> Dict[str, int]:
    """
    Returns a dictionary containing the sizes of the all the dimensions from the input PyTree.

    Args:
        input (PyTree[DataArray]): The input PyTree.
        dim_names (Sequence[str]): The names of the dimensions to retrieve the sizes for.

    Returns:
        Dict[str, int]: A dictionary mapping dimension names to their corresponding sizes.
    """
    input_sizes = tree_map(lambda x: dict(x.sizes), input)

    all_dims: Dict[str, int] = merge_dicts(input_sizes)

    return {d: all_dims[d] for d in dim_names}


def find_common_slices(data: DataArray, slices: Dict[str, slice]):
    common_keys = data.dims & slices.keys()
    return {k: slices[k] for k in common_keys}


def inplace_add_at_location(
    data: DataArray,
    location: Dict[str, slice],
    value: DataArray,
):
    data[find_common_slices(data, location)] += value


@overload
def infer(
    input_tree: PyTree[DataArray],
    output_size_tree: PyTree[Dict[str, int]],
    func: Callable,
    patch_size: Dict[str, int],
    overlap: Dict[str, float],
    filter_func: Callable,
    device: torch.device = torch.device("cpu"),
    merge_device: torch.device | XArrayDevice = "xarray",
    output_dtype=np.float32,
    *args,
    **kwargs,
):
    """currently beartype doesn't support deep dict typing"""
    patch_dims = patch_size.keys()
    # input_original_size = get_dim_size_dict_from_tree(input_tree, patch_dims)
    input_padded_tree, pad_sizes_tree = tree_transpose_map(
        lambda x: _pad_for_scaning_windows(x, patch_size, overlap),
        input_tree,
        inner_treespec=tree_structure((1, 1)),
    )
    pad_sizes = merge_dicts(pad_sizes_tree, _is_pad_size)
    input_padded_size = get_dim_size_dict_from_tree(input_padded_tree, patch_dims)
    patch_location = generate_patch_location(patch_size, input_padded_size, overlap)

    output_padded_size_tree = tree_map(
        # only update exsiting keys
        lambda x: {k: input_padded_size.get(k, v) for k, v in x.items()},
        output_size_tree,
        is_leaf=_is_named_size,
    )

    output_padded_tree = tree_map(
        lambda x: DataArray(
            data=np.zeros(tuple(x.values()), dtype=output_dtype), dims=x.keys()
        ),
        output_padded_size_tree,
        is_leaf=_is_named_size,
    )
    for loc in patch_location:
        data = tree_map(lambda x: x.isel(loc, missing_dims="ignore"), input_padded_tree)
        data_device = _transfer_to_device(data, device)
        result: PyTree[torch.Tensor] = func(data_device, *args, **kwargs)
        output_dims = tree_map(lambda x: x.dims, output_padded_tree)
        result_merge_device = _transfer_to_device(result, merge_device, output_dims)
        result_filtered = filter_func(result_merge_device, patch_size, overlap)
        output_padded_tree = tree_map_(
            lambda x, y: inplace_add_at_location(x, loc, y),
            output_padded_tree,
            result_filtered,
        )
    start_indices_tree = tree_map(
        lambda x: {k: pad_sizes.get(k, (0, 0))[0] for k in x.keys()},
        output_size_tree,
        is_leaf=_is_named_size,
    )
    output_cropped = crop(output_padded_tree, start_indices_tree, output_size_tree)
    ic(output_cropped[0].shape)
    return output_cropped


@dispatch
def infer(
    input,
    func,
    patch_dims,
    patch_size,
    overlap,
    split_func,
    filter_func,
    storage_device,
    device,
    *args,
    **kwargs,
):
    pass


@overload
def cutoff_filter(x: Tensor, dim: int, patch_size: int, overlap: float) -> Tensor:
    x = x.clone()
    step = int(patch_size * (1 - overlap))
    crop_start = int(patch_size * overlap) // 2
    start_location = [slice(None)] * x.dim()
    start_location[dim] = slice(0, crop_start)
    end_location = [slice(None)] * x.dim()
    end_location[dim] = slice(crop_start + step, None)
    x[start_location] = 0
    x[end_location] = 0
    return x


@overload
def cutoff_filter(
    x: Tensor,
    dims: Sequence[int],
    patch_size: Sequence[int],
    overlap: Sequence[float],
) -> Tensor:
    for d, p, o in zip(dims, patch_size, overlap):
        x = cutoff_filter(x, d, p, o)
    return x


def set_zero_(data, slices):
    for k, slc in find_common_slices(data, slices).items():
        data[{k: slc}] = 0
    # data[find_common_slices(data,front_slices)] = 0
    # data[find_common_slices(data,back_slices)] = 0


@overload
def cutoff_filter(
    data: PyTree[DataArray, "T"], patch_size: Dict[str, int], overlap: Dict[str, float]
) -> PyTree[DataArray, "T"]:
    data = tree_map(lambda x: x.copy(deep=True), data)
    front_cut_slices = tree_map(lambda x, y: slice(0, int(x * y)), patch_size, overlap)
    back_cut_slices = tree_map(
        lambda x, y: slice(int(x * (1 - y)), None), patch_size, overlap
    )
    tree_map_(lambda x: set_zero_(x, front_cut_slices), data)
    tree_map_(lambda x: set_zero_(x, back_cut_slices), data)
    return data
    # return x


@dispatch
def cutoff_filter(x, dims, patch_size, overlap):
    pass
