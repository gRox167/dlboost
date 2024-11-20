import torch


def get_coordinates(size, dim=2):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int"""
    tensors = [torch.linspace(-1, 1, s) for s in size]
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    return mgrid
