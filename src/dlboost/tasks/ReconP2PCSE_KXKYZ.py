# import re
# from typing import Any, Callable, Dict, Optional, Tuple, Union
import lightning.pytorch as pl
# from torchmetrics.image import TotalVariation
# from numpy import stack
# from sympy import cse
# import torch
# from torch import nn, norm
# from torch.nn import functional as f
# from torch import optim
# import einops as eo
# import pdb
# from matplotlib import pyplot as plt

# from monai.transforms import RandGridPatchd
# from monai.inferers import sliding_window_inference
# import torchkbnufft as tkbn

# from mrboost import computation as comp
# from dlboost.models.SpatialTransformNetwork import SpatialTransformNetwork
# from dlboost import losses
# from dlboost.utils import complex_as_real_2ch, real_2ch_as_complex, complex_as_real_ch, to_png
# from dlboost.tasks.boilerplate import *
# from dlboost.tasks.boilerplate_P2PCSE import *
from dlboost.tasks.boilerplate_P2PCSE_KXKYZ import P2PCSE_KXKYZ

class Recon(P2PCSE_KXKYZ):
    def __init__(self, nufft_im_size=..., patch_size=..., ch_pad=42, lr=0.0001, **kwargs):
        super().__init__(nufft_im_size, patch_size, ch_pad, lr, **kwargs)


