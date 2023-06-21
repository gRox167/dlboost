# import torch
import torch.nn as nn


activation_fn = {
     'relu': lambda: nn.ReLU(),
    'lrelu': lambda: nn.LeakyReLU(0.2),
    'gelu': lambda: nn.GELU(),
}

class ResBlock(nn.Module):
    def __init__(self, dimension, n_feats, kernel_size, bias=True, bn=False, act=nn.GELU(), res_scale=1):

        if dimension == 2:
            conv_fn = nn.Conv2d
            bn_fn = nn.BatchNorm2d
        elif dimension == 3:
            conv_fn = nn.Conv3d
            bn_fn = nn.BatchNorm3d
        else:
            raise ValueError()
        
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv_fn(n_feats, n_feats, kernel_size, padding=kernel_size // 2, bias=bias))
            if bn:
                m.append(bn_fn(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)#.mul(self.res_scale)
        res += x

        return res

class ConvBlock(nn.Module):
    """
    [conv_block] represents a single convolution block in the Unet which
    is a convolution based on the size of the input channel and output
    channels and then preforms a Leaky Relu with parameter 0.2.
    """

    def __init__(self, dim, in_channels, out_channels, stride=1):
        """
        Instiatiate the conv block
            :param dim: number of dimensions of the input
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: stride of the convolution
        """
        super().__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))

        padding = 1
        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 4
        else:
            raise Exception('stride must be 1 or 2')

        if dim == 3:
            stride = (1, stride, stride)
            ksize = 3
            padding = (ksize // 2, 1, 1)

        self.main = conv_fn(in_channels, out_channels, ksize, stride, padding)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Pass the input through the conv_block
        """
        out = self.main(x)
        out = self.activation(out)
        return out
