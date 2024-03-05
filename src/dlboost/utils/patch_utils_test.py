import torch

from dlboost.utils.patch_utils import cutoff_filter, infer, split_tensor


def test_one_tensor_patch_infer():
    input_tensor: torch.Tensor = torch.arange(1, 17).tile(1, 1, 16, 1)
    print(input_tensor.shape)
    func = lambda x: x
    output = infer(
        input_tensor,
        func,
        [2, 3],
        [4, 4],
        [0.5, 0.5],
        lambda x: split_tensor(x, [2, 3], [4, 4], [0.5, 0.5]),
        lambda x: cutoff_filter(x, [2, 3], [4, 4], [0.5, 0.5]),
    )
    assert torch.allclose(output, input_tensor)


def test_dict_tensor_patch_infer():
    input_dict = {
        "image": torch.arange(0, 16).tile(1, 1, 16, 1),
        "kspace": torch.arange(0, 32).tile(1, 15, 16, 1),
    }
    patch_dims = {
        "image": [2],
        "kspace": [2],
    }
    patch_sizes = {
        "image": [4],
        "kspace": [4],
    }
    overlap = {
        "image": [0.5],
        "kspace": [0.5],
    }
    func = lambda x: x["image"]
    output = infer(
        input_dict,
        func,
        patch_dims,
        patch_sizes,
        overlap,
        lambda x: split_tensor(x, patch_dims, patch_sizes, overlap),
        lambda x: cutoff_filter(x, [2], [4], [0.5]),
        "image",
    )
    assert torch.allclose(output, input_dict["image"])
