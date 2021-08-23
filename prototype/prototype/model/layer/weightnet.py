import torch.nn as nn
import torch.nn.functional as F

"""
    WeightNet: Revisiting the Design Space of Weight Networks
    https://arxiv.org/abs/2007.11823
"""


class WeightNet(nn.Module):
    r"""Applies WeightNet to a standard convolution.
    The grouped fc layer directly generates the convolutional kernel,
    this layer has M*inp inputs, G*oup groups and oup*inp*ksize*ksize outputs.
    M/G control the amount of parameters.

    Args:
        inp (int): Number of input channels
        oup (int): Number of output channels
        ksize (int): Size of the convolving kernel
        stride (int): Stride of the convolution
    """

    def __init__(self, inp, oup, ksize, stride):
        super().__init__()
        # control the number of groups in grouped FC
        self.M = 2
        self.G = 2

        self.pad = ksize // 2
        inp_gap = max(16, inp//16)
        self.inp = inp
        self.oup = oup
        self.ksize = ksize
        self.stride = stride

        # grouped fully connected operations (grouped FC)
        self.wn_fc1 = nn.Conv2d(inp_gap, self.M*oup, 1, 1, 0,
                                groups=1, bias=True)
        self.wn_fc2 = nn.Conv2d(self.M*oup, oup*inp*ksize*ksize,
                                1, 1, 0, groups=self.G*oup, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_gap):
        r""" Input:
            x (bs*c*h*w): output from previous convolution layer
            x_gap (bs*inp_gap*1*1): the output feature from reduction layer
        """

        # generate data dependent weights for standard convolution
        x_w = self.wn_fc1(x_gap)
        x_w = self.sigmoid(x_w)
        x_w = self.wn_fc2(x_w)

        batch_size = x.shape[0]
        if batch_size == 1:
            x_w = x_w.reshape(self.oup, self.inp, self.ksize, self.ksize)
            x = F.conv2d(x, weight=x_w, stride=self.stride, padding=self.pad)
            return x

        # 1*(bs*c)*kh*kw
        x = x.reshape(1, -1, x.shape[2], x.shape[3])
        # (bs*oc)*c*kh*kw
        x_w = x_w.reshape(-1, self.inp, self.ksize, self.ksize)
        x = F.conv2d(x, weight=x_w, stride=self.stride,
                     padding=self.pad, groups=batch_size)
        x = x.reshape(-1, self.oup, x.shape[2], x.shape[3])
        return x


class WeightNet_DW(nn.Module):
    r""" Here we show a grouping manner when we apply
    WeightNet to a depthwise convolution. The grouped
    fc layer directly generates the convolutional kernel,
    has fewer parameters while achieving comparable results.
    This layer has M/G*inp inputs, inp groups and inp*ksize*ksize outputs.

    Args:
        inp (int): Number of input channels
        oup (int): Number of output channels
        ksize (int): Size of the convolving kernel
        stride (int): Stride of the convolution
    """
    def __init__(self, inp, ksize, stride):
        super().__init__()
        # control the number of groups in grouped FC
        self.M = 2
        self.G = 2

        self.pad = ksize // 2
        inp_gap = max(16, inp//16)
        self.inp = inp
        self.ksize = ksize
        self.stride = stride

        # grouped fully connected operations (grouped FC)
        self.wn_fc1 = nn.Conv2d(inp_gap, self.M//self.G*inp,
                                1, 1, 0, groups=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.wn_fc2 = nn.Conv2d(self.M//self.G*inp, inp*ksize*ksize,
                                1, 1, 0, groups=inp, bias=False)

    def forward(self, x, x_gap):
        r""" Input:
            x (bs*c*h*w): the output feature from previous convolution layer
            x_gap (bs*inp_gap*1*1): the output feature from reduction layer
        """

        # generate data dependent weights for depth-wise convolution
        x_w = self.wn_fc1(x_gap)
        x_w = self.sigmoid(x_w)
        x_w = self.wn_fc2(x_w)

        batch_size = x.shape[0]
        x = x.reshape(1, -1, x.shape[2], x.shape[3])  # 1*(bs*c)*kh*kw
        x_w = x_w.reshape(-1, 1, self.ksize, self.ksize)  # (bs*oc)*1*kh*kw
        x = F.conv2d(x, weight=x_w, stride=self.stride,
                     padding=self.pad, groups=batch_size*self.inp)
        x = x.reshape(-1, self.inp, x.shape[2], x.shape[3])
        return x
