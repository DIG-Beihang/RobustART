import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from collections import OrderedDict


__all__ = ['hybird_resnet50_3stage', 'hybird_stem', 'hybird_resnet50_1stage']


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return StdConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.gn1 = nn.GroupNorm(32, planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.gn2 = nn.GroupNorm(32, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.gn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = StdConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(32, planes)
        self.conv2 = StdConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.gn2 = nn.GroupNorm(32, planes)
        self.conv3 = StdConv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.gn3 = nn.GroupNorm(32, planes * 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.gn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out


class ResNet(nn.Module):
    """Redidual Networks class, based on
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/abs/1512.03385>`_
    """
    def __init__(self,
                 block,
                 layers,
                 strides=[1, 2, 2, 2],
                 inplanes=64,
                 num_classes=1000,
                 deep_stem=False,
                 avg_down=False,
                 scale=1.0):
        r"""
        Arguments:

        - layers (:obj:`list` of 4 ints): how many layers in each stage
        - num_classes (:obj:`int`): number of classification classes[]
        - deep_stem (:obj:`bool`): whether to use deep_stem as the first conv
        - avg_down (:obj:`bool`): whether to use avg_down when spatial downsample
        """

        super(ResNet, self).__init__()

        self.inplanes = int(inplanes * scale)
        self.deep_stem = deep_stem
        self.avg_down = avg_down

        if self.deep_stem:
            self.conv1 = nn.Sequential(
                        StdConv2d(3, self.inplanes // 2, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.GroupNorm(32, self.inplanes // 2),
                        nn.ReLU(inplace=True),
                        StdConv2d(self.inplanes // 2, self.inplanes // 2, kernel_size=3,
                                  stride=1, padding=1, bias=False),
                        nn.GroupNorm(32, self.inplanes // 2),
                        nn.ReLU(inplace=True),
                        StdConv2d(self.inplanes // 2, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                    )
        else:
            self.conv1 = StdConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.gn1 = nn.GroupNorm(32, self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stages = OrderedDict()
        for i in range(len(layers)):
            stages['layer{}'.format(i + 1)] = self._make_layer(block, int(64 * scale), layers[i], stride=strides[i])
        self.layers = nn.Sequential(stages)

        for m in self.modules():
            if isinstance(m, StdConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0/float(n))
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, avg_down=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(stride, stride=stride, ceil_mode=True, count_include_pad=False),
                    StdConv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=1, bias=False),
                    nn.GroupNorm(32, planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    StdConv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(32, planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layers(x)
        return x


def hybird_resnet50_3stage(**kwargs):
    return ResNet(block=Bottleneck, layers=[3, 4, 9], **kwargs)


def hybird_stem(**kwargs):
    return ResNet(block=Bottleneck, layers=[], **kwargs)


def hybird_resnet50_1stage(**kwargs):
    return ResNet(block=Bottleneck, layers=[3], **kwargs)
