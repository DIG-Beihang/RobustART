import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from prototype.spring.linklink.nn import SyncBatchNorm2d
from prototype.prototype.utils.misc import get_bn


__all__ = ['bignas_resnet18_9M', 'bignas_resnet18_37M', 'bignas_resnet18_50M',
           'bignas_resnet18_49M', 'bignas_resnet18_66M', 'bignas_resnet18_1555M',
           'bignas_resnet18_107M', 'bignas_resnet18_125M', 'bignas_resnet18_150M',
           'bignas_resnet18_312M', 'bignas_resnet18_403M', 'bignas_resnet18_492M']


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


class Hswish(nn.Module):

    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


def build_activation(act_func, inplace=True):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func == 'h_swish':
        return Hswish(inplace=inplace)
    elif act_func == 'swish':
        return swish()
    elif act_func == 'h_sigmoid':
        return Hsigmoid(inplace=inplace)
    elif act_func is None:
        return None
    else:
        raise ValueError('do not support: %s' % act_func)


class LinearBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True, dropout_rate=0.):
        super(LinearBlock, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.dropout_rate = dropout_rate

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            self.dropout = None
        self.linear = nn.Linear(
            in_features=self.in_features, out_features=self.out_features, bias=self.bias
        )

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        return self.linear(x)


class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, dilation=1,
                 use_bn=True, act_func='relu'):
        super(ConvBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bn = use_bn
        self.act_func = act_func

        padding = get_same_padding(self.kernel_size)
        self.conv = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel,
                              kernel_size=self.kernel_size, padding=padding, groups=1,
                              stride=self.stride, dilation=self.dilation)
        if self.use_bn:
            self.bn = BN(self.out_channel)
        self.act = build_activation(self.act_func, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act(x)
        return x


class BottleneckBlock(nn.Module):

    def __init__(self, in_channel, out_channel,
                 kernel_size=3, expand_ratio=0.25, stride=1, act_func='relu'):
        super(BottleneckBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.kernel_size = kernel_size
        self.expand_ratio = expand_ratio

        self.stride = stride
        self.act_func = act_func

        # build modules
        middle_channel = make_divisible(self.out_channel * self.expand_ratio, 8)
        padding = get_same_padding(self.kernel_size)
        self.point_conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(self.in_channel, middle_channel, kernel_size=1, groups=1)),
            ('bn', nn.BatchNorm2d(middle_channel)),
            ('act', build_activation(self.act_func, inplace=True)),
        ]))

        self.normal_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(
                middle_channel, middle_channel, self.kernel_size, stride=self.stride, groups=1, padding=padding)),
            ('bn', nn.BatchNorm2d(middle_channel)),
            ('act', build_activation(self.act_func, inplace=True))
        ]))

        self.point_conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(middle_channel, self.out_channel, kernel_size=1, groups=1)),
            ('bn', nn.BatchNorm2d(self.out_channel)),
        ]))
        self.act3 = build_activation(self.act_func, inplace=True)

        if self.in_channel == self.out_channel and self.stride == 1:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, groups=1, stride=stride)
            self.shortcutbn = nn.BatchNorm2d(self.out_channel)

    def forward(self, x):
        identity = x

        x = self.point_conv1(x)
        x = self.normal_conv(x)
        x = self.point_conv2(x)
        if self.shortcut is None:
            x += identity
        else:
            x += self.shortcutbn(self.shortcut(identity))
        return self.act3(x)


class BasicBlock(nn.Module):

    def __init__(self, in_channel, out_channel,
                 kernel_size=3, expand_ratio=1, stride=1, act_func='relu'):
        super(BasicBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.kernel_size = kernel_size
        self.expand_ratio = expand_ratio

        self.stride = stride
        self.act_func = act_func

        # build modules default is 1
        middle_channel = make_divisible(self.out_channel * self.expand_ratio, 8)
        padding = get_same_padding(self.kernel_size)
        self.normal_conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(self.in_channel, middle_channel, self.kernel_size,
                               stride=self.stride, groups=1, padding=padding)),
            ('bn', BN(middle_channel)),
            ('act', build_activation(self.act_func, inplace=True))
        ]))

        self.normal_conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(middle_channel, self.out_channel, self.kernel_size, groups=1, padding=padding)),
            ('bn', BN(self.out_channel)),
        ]))
        self.act2 = build_activation(self.act_func, inplace=True)

        if self.in_channel == self.out_channel and self.stride == 1:
            self.shortcut = None
        else:
            self.shortcut = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, groups=1, stride=stride)),
                ('bn', BN(self.out_channel)),
            ]))

    def forward(self, x):
        identity = x

        x = self.normal_conv1(x)
        x = self.normal_conv2(x)
        if self.shortcut is None:
            x += identity
        else:
            x += self.shortcut(identity)
        return self.act2(x)


def get_same_length(element, depth):
    if len(element) == len(depth):
        element_list = []
        for i, d in enumerate(depth):
            element_list += [element[i]] * d
    elif len(element) == sum(depth):
        element_list = element
    else:
        raise ValueError('we only need stage-wise or block wise settings')
    return element_list


class BigNAS_ResNet_Basic(nn.Module):

    def __init__(self,
                 num_classes=1000,
                 width=[8, 8, 16, 48, 224],
                 depth=[1, 2, 2, 1, 2],
                 stride_stages=[2, 2, 2, 2, 2],
                 kernel_size=[7, 3, 3, 3, 3, 3, 3, 3],
                 expand_ratio=[0, 1, 1, 1, 1, 0.5, 0.5, 0.5],
                 act_stages=['relu', 'relu', 'relu', 'relu', 'relu'],
                 dropout_rate=0.,
                 bn=None):
        r"""
        Arguments:

        - num_classes (:obj:`int`): number of classification classes
        - width (:obj:`list` of 5 (stages+1) ints): channel list
        - depth (:obj:`list` of 5 (stages+1) ints): depth list for stages
        - stride_stages (:obj:`list` of 5 (stages+1) ints): stride list for stages
        - kernel_size (:obj:`list` of 8 (blocks+1) ints): kernel size list for blocks
        - expand_ratio (:obj:`list` of 8 (blocks+1) ints): expand ratio list for blocks
        - act_stages(:obj:`list` of 8 (blocks+1) ints): activation list for blocks
        - dropout_rate (:obj:`float`): dropout rate
        - bn (:obj:`dict`): definition of batchnorm
        """

        super(BigNAS_ResNet_Basic, self).__init__()

        global BN

        BN = get_bn(bn)

        self.depth = depth
        self.width = width
        self.kernel_size = get_same_length(kernel_size, self.depth)
        self.expand_ratio = get_same_length(expand_ratio, self.depth)

        self.dropout_rate = dropout_rate

        # first conv layer
        self.first_conv = ConvBlock(
            in_channel=3, out_channel=self.width[0], kernel_size=self.kernel_size[0],
            stride=stride_stages[0], act_func=act_stages[0])

        blocks = []
        _block_index = 0
        input_channel = self.width[0]

        stage_num = 1
        for s, act_func, n_block, output_channel in zip(stride_stages[1:], act_stages[1:], self.depth[1:],
                                                        self.width[1:]):
            _block_index += n_block
            kernel_size = self.kernel_size[_block_index]
            expand_ratio = self.expand_ratio[_block_index]
            stage_num += 1
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                basic_block = BasicBlock(
                        in_channel=input_channel, out_channel=output_channel, kernel_size=kernel_size,
                        expand_ratio=expand_ratio, stride=stride, act_func=act_func)
                blocks.append(basic_block)
                input_channel = output_channel

        self.blocks = nn.ModuleList(blocks)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.classifier = LinearBlock(
            in_features=self.width[-1], out_features=num_classes, bias=True, dropout_rate=dropout_rate)

        self.init_model()

    def forward(self, x):
        # first conv
        x = self.first_conv(x)

        # blocks
        for block in self.blocks:
            x = block(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def init_model(self):
        """ Conv2d, BatchNorm2d, BatchNorm1d, Linear, """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, SyncBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0 / float(n))
                if m.bias is not None:
                    m.bias.data.zero_()


def bignas_resnet18_9M(**kwargs):
    """
    equal to ResNet18-1/8
    """
    kwargs['width'] = [8, 8, 24, 32, 112]
    kwargs['depth'] = [1, 1, 1, 2, 1]
    kwargs['kernel_size'] = [3, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 1, 1]
    return BigNAS_ResNet_Basic(**kwargs)


def bignas_resnet18_37M(**kwargs):
    """
    equal to ResNet18-1/8
    """
    kwargs['width'] = [8, 8, 16, 48, 192]
    kwargs['depth'] = [1, 1, 1, 1, 2]
    kwargs['kernel_size'] = [7, 3, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 0.5, 0.5, 0.5]
    return BigNAS_ResNet_Basic(**kwargs)


def bignas_resnet18_49M(**kwargs):
    """
    equal to ResNet18-1/8
    """
    kwargs['width'] = [8, 8, 24, 32, 224]
    kwargs['depth'] = [1, 1, 1, 2, 2]
    kwargs['kernel_size'] = [7, 3, 3, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 1, 1, 0.25, 0.25]
    return BigNAS_ResNet_Basic(**kwargs)


def bignas_resnet18_50M(**kwargs):
    """
    equal to ResNet18-1/8
    """
    kwargs['width'] = [8, 8, 16, 48, 224]
    kwargs['depth'] = [1, 2, 2, 1, 2]
    kwargs['kernel_size'] = [7, 3, 3, 3, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 1, 1, 0.5, 0.5, 0.5]
    return BigNAS_ResNet_Basic(**kwargs)


def bignas_resnet18_66M(**kwargs):
    """
    equal to ResNet18-1/8
    """
    kwargs['width'] = [8, 8, 16, 48, 192]
    kwargs['depth'] = [1, 1, 1, 1, 2]
    kwargs['kernel_size'] = [7, 3, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 0.75, 0.75, 0.75]
    return BigNAS_ResNet_Basic(**kwargs)


def bignas_resnet18_1555M(**kwargs):
    """
    equal to ResNet18
    """
    kwargs['width'] = [32, 64, 112, 256, 592]
    kwargs['depth'] = [1, 1, 1, 3, 2]
    kwargs['kernel_size'] = [7, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 1, 1]
    return BigNAS_ResNet_Basic(**kwargs)


def bignas_resnet18_107M(**kwargs):
    """
    equal to ResNet18 1/4
    """
    kwargs['width'] = [16, 16, 32, 48, 160]
    kwargs['depth'] = [1, 1, 1, 2, 3]
    kwargs['kernel_size'] = [3, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 1, 1]
    return BigNAS_ResNet_Basic(**kwargs)


def bignas_resnet18_125M(**kwargs):
    """
    equal to ResNet18 1/4
    """
    kwargs['width'] = [16, 16, 48, 64, 192]
    kwargs['depth'] = [1, 1, 1, 2, 2]
    kwargs['kernel_size'] = [3, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 1, 1]
    return BigNAS_ResNet_Basic(**kwargs)


def bignas_resnet18_150M(**kwargs):
    """
    equal to ResNet18 1/4
    """
    kwargs['width'] = [16, 16, 48, 64, 192]
    kwargs['depth'] = [1, 1, 1, 1, 3]
    kwargs['kernel_size'] = [3, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 1, 1]
    return BigNAS_ResNet_Basic(**kwargs)


def bignas_resnet18_312M(**kwargs):
    """
    equal to ResNet18 1/2
    """
    kwargs['width'] = [24, 24, 48, 112, 320]
    kwargs['depth'] = [1, 1, 1, 2, 2]
    kwargs['kernel_size'] = [5, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 1, 1]
    return BigNAS_ResNet_Basic(**kwargs)


def bignas_resnet18_403M(**kwargs):
    """
    equal to ResNet18 1/2
    """
    kwargs['width'] = [16, 24, 48, 128, 320]
    kwargs['depth'] = [1, 1, 1, 2, 3]
    kwargs['kernel_size'] = [3, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 1, 1]
    return BigNAS_ResNet_Basic(**kwargs)


def bignas_resnet18_492M(**kwargs):
    """
    equal to ResNet18 1/2
    """
    kwargs['width'] = [32, 32, 64, 144, 320]
    kwargs['depth'] = [1, 1, 1, 2, 3]
    kwargs['kernel_size'] = [3, 3, 3, 3, 3]
    kwargs['expand_ratio'] = [0, 1, 1, 1, 1]
    return BigNAS_ResNet_Basic(**kwargs)


def bignas_resnet18(**kwargs):
    return BigNAS_ResNet_Basic(**kwargs)
