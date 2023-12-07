# import re
# from typing import Any, Callable, Dict, Optional, Tuple, Union
# import lightning.pytorch as pl
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
from dlboost.tasks.boilerplate_P2PCSE import *
from torch import nn, optim, Module

class Recon(P2PCSE):
    def __init__(self, recon_module: Module, cse_module: Module, nufft_im_size=..., patch_size=..., ch_pad=42, recon_loss_fn=nn.MSELoss, smooth_loss_coef=0.1, recon_optimizer=optim.Adam, recon_lr=0.0001, **kwargs):
        super().__init__(recon_module, cse_module, nufft_im_size, patch_size, ch_pad, recon_loss_fn, smooth_loss_coef, recon_optimizer, recon_lr, **kwargs)

