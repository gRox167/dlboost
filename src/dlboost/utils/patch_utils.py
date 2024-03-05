from typing import (
    Any,
    Callable,
    Dict,
    Sequence,
    Tuple,
)

import torch
from icecream import ic
from plum import dispatch, overload

from dlboost.utils.tensor_utils import _transfer_to_device, crop_tensor, pad_tensor
from dlboost.utils.type_utils import (
    Data_With_Location,
    Pad_Sizes,
)


def _pad_for_scaning_windows(
    input: torch.Tensor,
    patch_dims: Sequence[int],
    patch_size: Sequence[int],
    overlap: Sequence[float],
) -> Tuple[torch.Tensor, Pad_Sizes, torch.Size]:
    pad_sizes = [(round(s * o),) * 2 for s, o in zip(patch_size, overlap)]
    output = pad_tensor(input, patch_dims, pad_sizes, mode="constant", value=0)
    return output, pad_sizes, output.shape


@overload
def _forward_for_scanning_windows(
    input_splitted: Sequence[Data_With_Location],
    func: Callable,
    output: torch.Tensor,
    filter_func: Callable,
    # key_component: str,
    device: torch.device,
    storage_device: torch.device,
    *args: Any,
    **kwargs: Any,
):
    for data in input_splitted:
        d, l = _transfer_to_device(data, device)
        result = func(d, *args, **kwargs)
        output[l] += filter_func(result.to(storage_device))
    return output


@overload
def _forward_for_scanning_windows(
    input_splitted: Sequence[Dict[str, Data_With_Location]],
    func: Callable,
    output: torch.Tensor,
    filter_func: Callable,
    key_component: str,
    device: torch.device,
    storage_device: torch.device,
    *args: Any,
    **kwargs: Any,
):
    for data in input_splitted:
        l = data[key_component][1]
        d = _transfer_to_device(data, device)
        input_data = {k: v[0] for k, v in d.items()}
        result = func(input_data, *args, **kwargs)
        output[l] += filter_func(result.to(storage_device))
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
def infer(
    input: torch.Tensor,
    func: Callable,
    patch_dims: Sequence[int],
    patch_size: Sequence[int],
    overlap: Sequence[float],
    split_func: Callable[[torch.Tensor], Sequence[Data_With_Location]],
    filter_func: Callable,
    # batch_size: int,
    storage_device: torch.device = torch.device("cpu"),
    device: torch.device = torch.device("cpu"),
    *args,
    **kwargs,
):
    patch_dim_num = len(patch_dims)
    assert len(patch_size) == patch_dim_num and len(overlap) == patch_dim_num

    input_padded, pad_sizes, padded_size = _pad_for_scaning_windows(
        input, patch_dims, patch_size, overlap
    )

    input_splitted = split_func(input_padded)  # , patch_size, patch_dims, overlap)

    output = _forward_for_scanning_windows(
        input_splitted,
        func,
        torch.zeros(padded_size, device=storage_device, dtype=input.dtype),
        filter_func,
        device,
        storage_device,
        *args,
        **kwargs,
    )
    ic(output.shape, padded_size)
    output_cropped = crop_tensor(
        output,
        patch_dims,
        tuple(p1 for p1, p2 in pad_sizes),
        tuple(input.shape[d] for d in patch_dims),
    )
    return output_cropped


@overload
def infer(
    input: Dict[str, torch.Tensor],
    func: Callable,
    patch_dims: Dict[str, Sequence[int]],
    patch_size: Dict[str, Sequence[int]],
    overlap: Dict[str, Sequence[float]],
    split_func: Callable[
        [Dict[str, torch.Tensor]], Dict[str, Sequence[Data_With_Location]]
    ],
    filter_func: Callable,
    # batch_size: int,
    key_component: str,
    storage_device: torch.device = torch.device("cpu"),
    device: torch.device = torch.device("cpu"),
    *args,
    **kwargs,
):
    input_padded = dict()
    for k, v in input.items():
        if patch_dims[k] is None and patch_size[k] is None and overlap[k] is None:
            input_splitted[k] = [(v, (slice(None),) * v.dim())]
        else:
            if k == key_component:
                input_padded[k], pad_sizes, padded_size = _pad_for_scaning_windows(
                    v, patch_dims[k], patch_size[k], overlap[k]
                )
            else:
                input_padded[k], _, _ = _pad_for_scaning_windows(
                    v, patch_dims[k], patch_size[k], overlap[k]
                )
    input_splitted = split_func(input_padded)

    patch_num = len(input_splitted[key_component])
    input_splitted_T = [
        dict((k, v[i]) if len(v) != 1 else (k, v) for k, v in input_splitted.items())
        for i in range(patch_num)
    ]

    output = _forward_for_scanning_windows(
        input_splitted_T,
        func,
        torch.zeros(
            padded_size, device=storage_device, dtype=input[key_component].dtype
        ),
        filter_func,
        key_component,
        device,
        storage_device,
        *args,
        **kwargs,
    )

    output_cropped = crop_tensor(
        output,
        patch_dims[key_component],
        tuple(p1 for p1, p2 in pad_sizes),
        tuple(input[key_component].shape[d] for d in patch_dims[key_component]),
    )
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


@dispatch
def local_location_to_global_location(
    local_location: slice, global_location: Sequence[slice], dim: int
):
    """
    local slice: the slice index of the patch in the spcific dimension
    global slices: the slice index of the patch in the whole tensor,
    we can directly use the global slices to index the tensor to get patch
    """
    return tuple(
        local_location if d == dim else gl for d, gl in enumerate(global_location)
    )


@dispatch
def local_location_to_global_location(
    local_location: Sequence[slice], global_location: Sequence[slice], dim: int
):
    return tuple(
        local_location[d] if d == dim else gl for d, gl in enumerate(global_location)
    )


@overload
def split_1d(
    data: torch.Tensor,
    dim: int,
    patch_size: int,
    overlap: float = 0.5,
) -> Sequence[Data_With_Location]:
    total_dims = data.dim()
    step = int(patch_size * (1 - overlap))
    dimension_size = data.size(dim)
    local_locations = [
        slice(i, i + patch_size) for i in range(0, dimension_size - step, step)
    ]
    global_location = tuple(slice(None) for _ in range(total_dims))
    r = []
    for loc in local_locations:
        gs = local_location_to_global_location(loc, global_location, dim)
        r.append((data[gs], gs))
    return r


# @split_1d.register
@overload
def split_1d(
    data: Data_With_Location,
    dim: int,
    patch_size: int,
    overlap: float = 0.5,
) -> Sequence[Data_With_Location]:
    """
    each dimension can only be splitted once
    """
    _data, global_slices = data
    splitted_data_and_location = split_1d(_data, dim, patch_size, overlap)
    return [
        (d, local_location_to_global_location(loc, global_slices, dim))
        for d, loc in splitted_data_and_location
    ]


# @split_1d.register
@overload
def split_1d(
    data: Sequence[Data_With_Location],
    dim: int,
    patch_size: int,
    overlap: float = 0.5,
) -> Sequence[Data_With_Location]:
    l = [split_1d(d, dim, patch_size, overlap) for d in data]
    return [item for sublist in l for item in sublist]


@dispatch
def split_1d(data, dim, patch_size, overlap=0.5):
    pass


@overload
def split_tensor(
    data: torch.Tensor,
    dims: Sequence[int],
    patch_size: Sequence[int],
    overlap: Sequence[float],
) -> Sequence[Data_With_Location]:
    for d, p, o in zip(dims, patch_size, overlap):
        data = split_1d(data, d, p, o)
    return data


@overload
def split_tensor(
    data: Dict[str, torch.Tensor],
    dims: Dict[str, Sequence[int]],
    patch_size: Dict[str, Sequence[int]],
    overlap: Dict[str, Sequence[float]],
) -> Dict[str, Sequence[Data_With_Location]]:
    return {
        k: split_tensor(data[k], dims[k], patch_size[k], overlap[k])
        for k in data.keys()
    }


@dispatch
def split_tensor(data, dims, patch_size, overlap):
    pass


@overload
def cutoff_filter(
    x: torch.Tensor, dim: int, patch_size: int, overlap: int
) -> torch.Tensor:
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
    x: torch.Tensor,
    dims: Sequence[int],
    patch_size: Sequence[int],
    overlap: Sequence[int],
) -> torch.Tensor:
    for d, p, o in zip(dims, patch_size, overlap):
        x = cutoff_filter(x, d, p, o)
    return x


@dispatch
def cutoff_filter(x, dims, patch_size, overlap):
    pass


# from monai.config import IndexSelection, KeysCollection
# from monai.config.type_definitions import NdarrayOrTensor
# from monai.data.utils import (
#     compute_importance_map,
#     get_valid_patch_size,
# )
# from monai.inferers import Inferer
# from monai.transforms import Resize
# from monai.transforms.croppad.array import (
#     RandCropByPosNegLabel,
#     SpatialCrop,
#     SpatialPad,
# )
# from monai.transforms.inverse import InvertibleTransform
# from monai.transforms.transform import MapTransform, Randomizable
# from monai.transforms.utils import (
#     generate_pos_neg_label_crop_centers,
#     map_binary_to_indices,
# )
# from monai.utils import (
#     BlendMode,
#     PytorchPadMode,
#     ensure_tuple,
#     ensure_tuple_rep,
#     ensure_tuple_size,
#     fall_back_tuple,
#     first,
#     look_up_option,
# )
# from monai.utils import ImageMetaKey as Key
# class PyramidRandCropByPosNegLabeld(Randomizable, MapTransform, InvertibleTransform):
#     backend = RandCropByPosNegLabel.backend

#     def __init__(
#         self,
#         keys: KeysCollection,
#         label_key: str,
#         spatial_size: Union[Sequence[int], int],
#         pyramid_size: int = 3,
#         pos: float = 1.0,
#         neg: float = 1.0,
#         num_samples: int = 1,
#         image_key: Optional[str] = None,
#         image_threshold: float = 0.0,
#         fg_indices_key: Optional[str] = None,
#         bg_indices_key: Optional[str] = None,
#         meta_keys: Optional[KeysCollection] = None,
#         meta_key_postfix: str = "meta_dict",
#         allow_smaller: bool = False,
#         allow_missing_keys: bool = False,
#     ) -> None:
#         MapTransform.__init__(self, keys, allow_missing_keys)
#         self.label_key = label_key
#         self.spatial_size: Union[Tuple[int, ...], Sequence[int], int] = spatial_size
#         self.pyramid_size: int = pyramid_size
#         if pos < 0 or neg < 0:
#             raise ValueError(
#                 f"pos and neg must be nonnegative, got pos={pos} neg={neg}."
#             )
#         if pos + neg == 0:
#             raise ValueError("Incompatible values: pos=0 and neg=0.")
#         self.pos_ratio = pos / (pos + neg)
#         self.num_samples = num_samples
#         self.image_key = image_key
#         self.image_threshold = image_threshold
#         self.fg_indices_key = fg_indices_key
#         self.bg_indices_key = bg_indices_key
#         self.meta_keys = (
#             ensure_tuple_rep(None, len(self.keys))
#             if meta_keys is None
#             else ensure_tuple(meta_keys)
#         )
#         if len(self.keys) != len(self.meta_keys):
#             raise ValueError("meta_keys should have the same length as keys.")
#         self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
#         self.centers: Optional[List[List[int]]] = None
#         self.allow_smaller = allow_smaller

#     def randomize(
#         self,
#         label: NdarrayOrTensor,
#         fg_indices: Optional[NdarrayOrTensor] = None,
#         bg_indices: Optional[NdarrayOrTensor] = None,
#         image: Optional[NdarrayOrTensor] = None,
#     ) -> None:
#         self.spatial_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
#         if fg_indices is None or bg_indices is None:
#             fg_indices_, bg_indices_ = map_binary_to_indices(
#                 label, image, self.image_threshold
#             )
#         else:
#             fg_indices_ = fg_indices
#             bg_indices_ = bg_indices
#         self.centers = generate_pos_neg_label_crop_centers(
#             self.spatial_size,
#             self.num_samples,
#             self.pos_ratio,
#             label.shape[1:],
#             fg_indices_,
#             bg_indices_,
#             self.R,
#             # self.allow_smaller,
#         )

#     def __call__(
#         self, data: Mapping[Hashable, NdarrayOrTensor]
#     ) -> List[Dict[Hashable, NdarrayOrTensor]]:
#         d = dict(data)
#         label = d[self.label_key]
#         image = d[self.image_key] if self.image_key else None
#         fg_indices = (
#             d.pop(self.fg_indices_key, None)
#             if self.fg_indices_key is not None
#             else None
#         )
#         bg_indices = (
#             d.pop(self.bg_indices_key, None)
#             if self.bg_indices_key is not None
#             else None
#         )

#         self.randomize(label, fg_indices, bg_indices, image)
#         if not isinstance(self.spatial_size, tuple):
#             raise ValueError("spatial_size must be a valid tuple.")
#         if self.centers is None:
#             raise ValueError("no available ROI centers to crop.")

#         spatial_size_list = [
#             np.array(self.spatial_size) * 2**i for i in range(self.pyramid_size)
#         ]
#         # initialize returned list with shallow copy to preserve key ordering
#         results: List[Dict[Hashable, NdarrayOrTensor]] = [
#             dict(d) for _ in range(self.num_samples)
#         ]

#         for i, center in enumerate(self.centers):
#             # fill in the extra keys with unmodified data
#             for key in set(d.keys()).difference(set(self.keys)):
#                 results[i][key] = deepcopy(d[key])
#             for key in self.key_iterator(d):
#                 img = d[key]
#                 orig_size = img.shape[1:]

#                 # pad image to the large ROI
#                 pad_size = [orig_size[j] + spatial_size_list[-1][j] for j in range(3)]
#                 padder = SpatialPad(pad_size)
#                 padded_img = padder(img)

#                 # recenter the center
#                 img_list = []
#                 pad_center = [
#                     center[j] + spatial_size_list[-1][j] // 2 for j in range(3)
#                 ]
#                 if key == "label" or key == "mask":
#                     resizer = Resize(spatial_size_list[0], mode="nearest")
#                 else:
#                     resizer = Resize(
#                         spatial_size_list[0], mode="trilinear", align_corners=True
#                     )
#                 for size in spatial_size_list:
#                     cropper = SpatialCrop(roi_center=pad_center, roi_size=size)
#                     cropped = cropper(padded_img)
#                     resized = resizer(cropped)
#                     img_list.append(resized)
#                 results[i][key] = torch.cat(img_list, dim=0)
#                 self.push_transform(
#                     results[i], key, extra_info={"center": center}, orig_size=orig_size
#                 )
#             # add `patch_index` to the meta data
#             for key, meta_key, meta_key_postfix in self.key_iterator(
#                 d, self.meta_keys, self.meta_key_postfix
#             ):
#                 meta_key = meta_key or f"{key}_{meta_key_postfix}"
#                 if meta_key not in results[i]:
#                     results[i][meta_key] = {}  # type: ignore
#                 results[i][meta_key][Key.PATCH_INDEX] = i  # type: ignore

#         return results


# def pyramid_sliding_window_inference(
#     inputs: torch.Tensor,
#     roi_size: Union[Sequence[int], int],
#     sw_batch_size: int,
#     predictor: Callable[..., torch.Tensor],
#     pyramid_size: int = 3,
#     overlap: float = 0.25,
#     mode: Union[BlendMode, str] = BlendMode.CONSTANT,
#     sigma_scale: Union[Sequence[float], float] = 0.125,
#     padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
#     cval: float = 0.0,
#     sw_device: Union[torch.device, str, None] = None,
#     device: Union[torch.device, str, None] = None,
#     *args: Any,
#     **kwargs: Any,
# ) -> torch.Tensor:
#     """
#     Sliding window inference on `inputs` with `predictor`.

#     When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
#     To maintain the same spatial sizes, the output image will be cropped to the original input size.

#     Args:
#         inputs: input image to be processed (assuming NCHW[D])
#         roi_size: the spatial window size for inferences.
#             When its components have None or non-positives, the corresponding inputs dimension will be used.
#             if the components of the `roi_size` are non-positive values, the transform will use the
#             corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
#             to `(32, 64)` if the second spatial dimension size of img is `64`.
#         sw_batch_size: the batch size to run window slices.
#         predictor: given input tensor `patch_data` in shape NCHW[D], `predictor(patch_data)`
#             should return a prediction with the same spatial shape and batch_size, i.e. NMHW[D];
#             where HW[D] represents the patch spatial size, M is the number of output channels, N is `sw_batch_size`.
#         overlap: Amount of overlap between scans.
#         mode: {``"constant"``, ``"gaussian"``}
#             How to blend output of overlapping windows. Defaults to ``"constant"``.

#             - ``"constant``": gives equal weight to all predictions.
#             - ``"gaussian``": gives less weight to predictions on edges of windows.

#         sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
#             Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
#             When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
#             spatial dimensions.
#         padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
#             Padding mode for ``inputs``, when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
#             See also: https://pytorch.org/docs/stable/nn.functional.html#pad
#         cval: fill value for 'constant' padding mode. Default: 0
#         sw_device: device for the window data.
#             By default the device (and accordingly the memory) of the `inputs` is used.
#             Normally `sw_device` should be consistent with the device where `predictor` is defined.
#         device: device for the stitched output prediction.
#             By default the device (and accordingly the memory) of the `inputs` is used. If for example
#             set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
#             `inputs` and `roi_size`. Output is on the `device`.
#         args: optional args to be passed to ``predictor``.
#         kwargs: optional keyword args to be passed to ``predictor``.

#     Note:
#         - input must be channel-first and have a batch dim, supports N-D sliding window.

#     """
#     num_spatial_dims = len(inputs.shape) - 2
#     if overlap < 0 or overlap >= 1:
#         raise AssertionError("overlap must be >= 0 and < 1.")

#     # determine image spatial size and batch size
#     # Note: all input images must have the same image size and batch size
#     image_size_ = list(inputs.shape[2:])
#     batch_size = inputs.shape[0]

#     if device is None:
#         device = inputs.device
#     if sw_device is None:
#         sw_device = inputs.device

#     roi_size = fall_back_tuple(roi_size, image_size_)
#     roi_size_max = tuple(i * 2 ** (pyramid_size - 1) for i in roi_size)
#     # in case that image size is smaller than roi size
#     image_size = tuple(
#         max(image_size_[i], roi_size_max[i] + image_size_[i])
#         for i in range(num_spatial_dims)
#     )
#     pad_size = []
#     for k in range(len(inputs.shape) - 1, 1, -1):
#         # pad
#         diff = roi_size_max[k - 2]
#         half = diff // 2
#         pad_size.extend([half, diff - half])
#     inputs = F.pad(
#         inputs,
#         pad=pad_size,
#         mode=look_up_option(padding_mode, PytorchPadMode).value,
#         value=cval,
#     )

#     scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

#     # Store all slices in list
#     # slices = dense_patch_slices(image_size, roi_size, scan_interval)
#     roi_size_pyramid = [np.array(roi_size) * 2**i for i in range(pyramid_size)]
#     slices_pyramid = pyramid_dense_patch_slices(image_size, roi_size, scan_interval, 3)
#     num_win = len(slices_pyramid[0])  # number of windows per image
#     total_slices = num_win * batch_size  # total number of windows

#     # Create window-level importance map
#     importance_map = compute_importance_map(
#         get_valid_patch_size(image_size, roi_size),
#         mode=mode,
#         sigma_scale=sigma_scale,
#         device=device,
#     )

#     # Perform predictions
#     output_image, count_map = (
#         torch.tensor(0.0, device=device),
#         torch.tensor(0.0, device=device),
#     )
#     _initialized = False
#     for slice_g in range(0, total_slices, sw_batch_size):
#         slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
#         window_data_pyramid = []
#         for p in range(pyramid_size):
#             slices = slices_pyramid[p]
#             unravel_slice_ = [
#                 [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)]
#                 + list(slices[idx % num_win])
#                 for idx in slice_range
#             ]
#             if p == 0:
#                 # store patches coordinate location
#                 unravel_slice = unravel_slice_
#             patch_data = [
#                 F.interpolate(
#                     inputs[win_slice],
#                     size=roi_size,
#                     mode="trilinear",
#                     align_corners=True,
#                 )
#                 for win_slice in unravel_slice_
#             ]
#             window_data_pyramid.append(torch.cat(patch_data))
#         window_data = torch.cat(window_data_pyramid, dim=1).to(sw_device)
#         # batched patch segmentation
#         seg_prob = predictor(window_data, *args, **kwargs)[0].to(device)

#         if not _initialized:  # init. buffer at the first iteration
#             output_classes = seg_prob.shape[1]
#             output_shape = [batch_size, output_classes] + list(image_size)
#             # allocate memory to store the full output and the count for overlapping parts
#             output_image = torch.zeros(output_shape, dtype=torch.float32, device=device)
#             count_map = torch.zeros(output_shape, dtype=torch.float32, device=device)
#             _initialized = True

#         # store the result in the proper location of the full output. Apply weights from importance map.
#         for idx, original_idx in zip(slice_range, unravel_slice):
#             output_image[original_idx] += importance_map * seg_prob[idx - slice_g]
#             count_map[original_idx] += importance_map

#     # account for any overlapping sections
#     output_image = output_image / count_map

#     final_slicing: List[slice] = []
#     for sp in range(num_spatial_dims):
#         slice_dim = slice(
#             pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2]
#         )
#         final_slicing.insert(0, slice_dim)
#     while len(final_slicing) < len(output_image.shape):
#         final_slicing.insert(0, slice(None))
#     return output_image[final_slicing]


# class PyramidSlidingWindowInferer(Inferer):
#     """
#     Sliding window method for model inference,
#     with `sw_batch_size` windows for every model.forward().
#     Usage example can be found in the :py:class:`monai.inferers.Inferer` base class.

#     Args:
#         roi_size: the window size to execute SlidingWindow evaluation.
#             If it has non-positive components, the corresponding `inputs` size will be used.
#             if the components of the `roi_size` are non-positive values, the transform will use the
#             corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
#             to `(32, 64)` if the second spatial dimension size of img is `64`.
#         sw_batch_size: the batch size to run window slices.
#         overlap: Amount of overlap between scans.
#         mode: {``"constant"``, ``"gaussian"``}
#             How to blend output of overlapping windows. Defaults to ``"constant"``.

#             - ``"constant``": gives equal weight to all predictions.
#             - ``"gaussian``": gives less weight to predictions on edges of windows.

#         sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
#             Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
#             When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
#             spatial dimensions.
#         padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
#             Padding mode when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
#             See also: https://pytorch.org/docs/stable/nn.functional.html#pad
#         cval: fill value for 'constant' padding mode. Default: 0
#         sw_device: device for the window data.
#             By default the device (and accordingly the memory) of the `inputs` is used.
#             Normally `sw_device` should be consistent with the device where `predictor` is defined.
#         device: device for the stitched output prediction.
#             By default the device (and accordingly the memory) of the `inputs` is used. If for example
#             set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
#             `inputs` and `roi_size`. Output is on the `device`.

#     Note:
#         ``sw_batch_size`` denotes the max number of windows per network inference iteration,
#         not the batch size of inputs.

#     """

#     def __init__(
#         self,
#         roi_size: Union[Sequence[int], int],
#         sw_batch_size: int = 1,
#         overlap: float = 0.25,
#         pyramid_size: int = 3,
#         mode: Union[BlendMode, str] = BlendMode.CONSTANT,
#         sigma_scale: Union[Sequence[float], float] = 0.125,
#         padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
#         cval: float = 0.0,
#         sw_device: Union[torch.device, str, None] = None,
#         device: Union[torch.device, str, None] = None,
#     ) -> None:
#         Inferer.__init__(self)
#         self.roi_size = roi_size
#         self.sw_batch_size = sw_batch_size
#         self.overlap = overlap
#         self.mode: BlendMode = BlendMode(mode)
#         self.sigma_scale = sigma_scale
#         self.padding_mode = padding_mode
#         self.cval = cval
#         self.sw_device = sw_device
#         self.device = device
#         self.pyramid_size = pyramid_size

#     def __call__(
#         self,
#         inputs: torch.Tensor,
#         network: Callable[..., torch.Tensor],
#         *args: Any,
#         **kwargs: Any,
#     ) -> torch.Tensor:
#         """

#         Args:
#             inputs: model input data for inference.
#             network: target model to execute inference.
#                 supports callables such as ``lambda x: my_torch_model(x, additional_config)``
#             args: optional args to be passed to ``network``.
#             kwargs: optional keyword args to be passed to ``network``.

#         """
#         return pyramid_sliding_window_inference(
#             inputs,
#             self.roi_size,
#             self.sw_batch_size,
#             network,
#             self.pyramid_size,
#             self.overlap,
#             self.mode,
#             self.sigma_scale,
#             self.padding_mode,
#             self.cval,
#             self.sw_device,
#             self.device,
#             *args,
#             **kwargs,
#         )


# def _get_scan_interval(
#     image_size: Sequence[int],
#     roi_size: Sequence[int],
#     num_spatial_dims: int,
#     overlap: float,
# ) -> Tuple[int, ...]:
#     """
#     Compute scan interval according to the image size, roi size and overlap.
#     Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
#     use 1 instead to make sure sliding window works.
#     """
#     if len(image_size) != num_spatial_dims:
#         raise ValueError("image coord different from spatial dims.")
#     if len(roi_size) != num_spatial_dims:
#         raise ValueError("roi coord different from spatial dims.")

#     scan_interval = []
#     for i in range(num_spatial_dims):
#         if roi_size[i] == image_size[i]:
#             scan_interval.append(int(roi_size[i]))
#         else:
#             interval = int(roi_size[i] * (1 - overlap))
#             scan_interval.append(interval if interval > 0 else 1)
#     return tuple(scan_interval)


# def pyramid_dense_patch_slices(
#     image_size: Sequence[int],
#     patch_size: Sequence[int],
#     scan_interval: Sequence[int],
#     pyramid_size: int,
# ) -> List[Tuple[slice, ...]]:
#     """
#     Enumerate all slices defining ND patches of size `patch_size` from an `image_size` input image.

#     Args:
#         image_size: dimensions of image to iterate over
#         patch_size: size of patches to generate slices
#         scan_interval: dense patch sampling interval

#     Returns:
#         a list of slice objects defining each patch

#     """
#     num_spatial_dims = len(image_size)
#     patch_size = get_valid_patch_size(image_size, patch_size)
#     scan_interval = ensure_tuple_size(scan_interval, num_spatial_dims)
#     patch_size_pyramid = [np.array(patch_size) * 2**i for i in range(pyramid_size)]

#     scan_num = []
#     for i in range(num_spatial_dims):
#         if scan_interval[i] == 0:
#             scan_num.append(1)
#         else:
#             num = int(
#                 math.ceil(
#                     float(
#                         (image_size[i] - patch_size_pyramid[-1][i]) / scan_interval[i]
#                     )
#                 )
#             )
#             scan_dim = first(
#                 d
#                 for d in range(num)
#                 if d * scan_interval[i] + patch_size[i]
#                 >= image_size[i] - patch_size_pyramid[-1][i]
#             )
#             scan_num.append(scan_dim + 1 if scan_dim is not None else 1)

#     starts = []
#     for dim in range(num_spatial_dims):
#         dim_starts = []
#         for idx in range(scan_num[dim]):
#             start_idx = idx * scan_interval[dim] + patch_size_pyramid[-1][dim] // 2
#             start_idx -= max(start_idx + patch_size[dim] - image_size[dim], 0)
#             dim_starts.append(start_idx)
#         starts.append(dim_starts)

#     out = np.asarray([x.flatten() for x in np.meshgrid(*starts, indexing="ij")]).T
#     out_pyramid = [out]
#     for p in range(1, pyramid_size):
#         size_cur = patch_size_pyramid[p]
#         out_pyramid.append(
#             np.stack(
#                 [
#                     out[:, i] - (size_cur[i] - patch_size[i]) // 2
#                     for i in range(num_spatial_dims)
#                 ],
#                 axis=1,
#             )
#         )
#     return [
#         [tuple(slice(s, s + size[d]) for d, s in enumerate(x)) for x in level]
#         for level, size in zip(out_pyramid, patch_size_pyramid)
#     ]

# input_tensor = torch.arange(0, 16).tile(2, 1, 16, 1)
# print(input_tensor.shape)
# inferer = PyramidSlidingWindowInferer((2, 2), 2, 0.5, 3)
# output = inferer(input_tensor.float(), torch.nn.Identity())
# print(output.shape)
