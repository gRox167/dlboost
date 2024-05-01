import numpy as np
import torch
from icecream import ic
from xarray import DataArray

from dlboost.utils.patch_utils import cutoff_filter, infer, split_tensor
from dlboost.utils.tensor_utils import _transfer_to_device


def test_pytree_xarray_kspace_to_image_patch_infer():
    input_tree = [
        DataArray(dims = ("b", "ch", "h", "w"), data = np.tile(np.arange(0, 8, dtype=float), (1, 2, 6, 1))),
        DataArray(dims = ("b", "ch", "sp", "len"), data = np.tile(np.arange(0, 32, dtype=float), (1, 2, 3, 1))),
    ]
    output_sizes_tree = {"b": 1, "ch": 2, "h": 6, "w": 8}
        # {"b": 1, "ch": 2, "sp": 7, "len": 32},
    
    patch_sizes = {
        "ch": 1,
        "h": 4,
        "w": 4,
        # "len": 4,
    }
    overlap = {
        "ch": 0.0,
        "h": 0.25,
        "w": 0.25,
        # "len": 0.25,
    }

    func = lambda x: x[0]
    output = infer(
        input_tree,
        output_sizes_tree,
        func,
        patch_sizes,
        overlap,
        cutoff_filter,
    )
    assert torch.allclose(_transfer_to_device(output, torch.device("cpu")), _transfer_to_device(input_tree[0], torch.device("cpu")))
    # assert torch.allclose(_transfer_to_device(output[1], torch.device("cpu")), _transfer_to_device(input_tree[1], torch.device("cpu")))

def test_pytree_xarray_image_patch_infer():
    input_tree = DataArray(dims = ("b", "ch", "h", "w"), data = np.tile(np.arange(0, 8, dtype=float), (1, 2, 6, 1)))
    
    output_sizes_tree = {"b": 1, "ch": 2, "h": 6, "w": 8}
    
    patch_sizes = {
        "h": 4,
        "w": 4,
        # "len": 4,
    }
    overlap = {
        "h": 0.25,
        "w": 0.25,
        # "len": 0.25,
    }

    func = lambda x: x
    output = infer(
        input_tree,
        output_sizes_tree,
        func,
        patch_sizes,
        overlap,
        cutoff_filter,
    )
    assert torch.allclose(_transfer_to_device(output, torch.device("cpu")), _transfer_to_device(input_tree, torch.device("cpu")))

# def test_one_tensor_patch_infer():
#     input_tensor: torch.Tensor = torch.arange(1, 17).tile(1, 1, 16, 1)
#     print(input_tensor.shape)
#     func = lambda x: x
#     output = infer(
#         input_tensor,
#         func,
#         [2, 3],
#         [4, 4],
#         [0.5, 0.5],
#         split_tensor,
#         cutoff_filter,
#     )
#     assert torch.allclose(output, input_tensor)


# def test_dict_tensor_patch_infer():
#     input_dict = {
#         "image": torch.arange(0, 16).tile(1, 1, 16, 1),
#         "kspace": torch.arange(0, 32).tile(1, 15, 16, 1),
#     }
#     patch_dims = {
#         "image": [2],
#         "kspace": [2],
#     }
#     patch_sizes = {
#         "image": [4],
#         "kspace": [4],
#     }
#     overlap = {
#         "image": [0.5],
#         "kspace": [0.5],
#     }
#     func = lambda x: x["image"]
#     output = infer(
#         input_dict,
#         func,
#         patch_dims,
#         patch_sizes,
#         overlap,
#         split_tensor,
#         cutoff_filter,
#         "image",
#     )
#     assert torch.allclose(output, input_dict["image"])
