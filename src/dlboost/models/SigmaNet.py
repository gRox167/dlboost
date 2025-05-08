import torch
from torch import nn

from dlboost.models.DWUNet import DWInvertedBlock


class SigmaNet(nn.Module):
    r"""SigmaNet: a neural network to estimate the noise level.
    The network is a simple convnet encoder with several nD convolutional layers and a final pooling layer.
    """

    def __init__(
        self,
        in_channels=10,
        out_channels=1,
        features=(16, 32, 64, 128, 256),
        strides=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        spatial_dims=3,
    ):
        super().__init__()
        self.conv1 = DWInvertedBlock(
            spatial_dims,
            in_channels,
            features[0],
            stride=strides[0],
            kernel_size=3,
            norm=("instance", {"affine": True}),
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        )
        self.conv2 = DWInvertedBlock(
            spatial_dims,
            features[0],
            features[1],
            stride=strides[1],
            kernel_size=3,
            norm=("instance", {"affine": True}),
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        )
        self.conv3 = DWInvertedBlock(
            spatial_dims,
            features[1],
            features[2],
            stride=strides[2],
            kernel_size=3,
            norm=("instance", {"affine": True}),
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        )
        self.pooling = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(features[2], out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
