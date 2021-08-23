import torch
import torch.nn as nn
import torch.nn.functional as F

import prototype.spring.linklink as link
from prototype.prototype.utils.misc import get_bn

__all__ = ['mobilenet_v3']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_bn(inp, oup, stride, activation=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        BN(oup),
        activation(inplace=True)
    )


def conv_1x1_bn(inp, oup, activation=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        BN(oup),
        activation(inplace=True)
    )


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


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        if nl == 'RE':
            activation = nn.ReLU
        elif nl == 'HS':
            activation = Hswish
        else:
            raise NotImplementedError

        SELayer = SEModule if se else Identity

        layers = []
        if inp != exp:
            # pw
            layers.extend([
                nn.Conv2d(inp, exp, 1, 1, 0, bias=False),
                BN(exp),
                activation(inplace=True),
            ])
        layers.extend([
            # dw
            nn.Conv2d(exp, exp, kernel, stride,
                      padding, groups=exp, bias=False),
            BN(exp),
            SELayer(exp),
            activation(inplace=True),
            # pw-linear
            nn.Conv2d(exp, oup, 1, 1, 0, bias=False),
            BN(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    """
    MobileNet V3 main class, based on
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_
    """
    def __init__(self,
                 num_classes=1000,
                 scale=1.0,
                 dropout=0.8,
                 round_nearest=8,
                 mode='small',
                 bn=None):
        r"""
        Arguments:
            - num_classes (:obj:`int`): Number of classes
            - scale (:obj:`float`): Width multiplier, adjusts number of channels in each layer by this amount
            - dropout (:obj:`float`): Dropout rate
            - round_nearest (:obj:`int`): Round the number of channels in each layer to be a multiple of this number
              Set to 1 to turn off rounding
            - mode (:obj:`string`): model type, 'samll' or 'large'
            - bn (:obj:`dict`): definition of batchnorm
        """
        super(MobileNetV3, self).__init__()

        global BN
        BN = get_bn(bn)

        input_channel = 16
        last_channel = 1280
        if mode == 'large':
            mobile_setting = [
                [3, 16,  16,  False, 'RE', 1],
                [3, 64,  24,  False, 'RE', 2],
                [3, 72,  24,  False, 'RE', 1],
                [5, 72,  40,  True,  'RE', 2],
                [5, 120, 40,  True,  'RE', 1],
                [5, 120, 40,  True,  'RE', 1],
                [3, 240, 80,  False, 'HS', 2],
                [3, 200, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 480, 112, True,  'HS', 1],
                [3, 672, 112, True,  'HS', 1],
                [5, 672, 160, True,  'HS', 2],
                [5, 960, 160, True,  'HS', 1],
                [5, 960, 160, True,  'HS', 1],
            ]
        elif mode == 'small':
            mobile_setting = [
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [5, 96,  40,  True,  'HS', 2],
                [5, 240, 40,  True,  'HS', 1],
                [5, 240, 40,  True,  'HS', 1],
                [5, 120, 48,  True,  'HS', 1],
                [5, 144, 48,  True,  'HS', 1],
                [5, 288, 96,  True,  'HS', 2],
                [5, 576, 96,  True,  'HS', 1],
                [5, 576, 96,  True,  'HS', 1],
            ]
        else:
            raise NotImplementedError

        # building first layer
        last_channel = _make_divisible(
            last_channel * scale, round_nearest) if scale > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, activation=Hswish)]
        self.classifier = []

        # building mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = _make_divisible(c * scale, round_nearest)
            exp_channel = _make_divisible(exp * scale, round_nearest)
            self.features.append(InvertedResidual(
                input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # building last several layers
        if mode == 'large':
            last_conv = _make_divisible(960 * scale, round_nearest)
            self.features.append(conv_1x1_bn(
                input_channel, last_conv, activation=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        elif mode == 'small':
            last_conv = _make_divisible(576 * scale, round_nearest)
            self.features.append(conv_1x1_bn(
                input_channel, last_conv, activation=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        else:
            raise NotImplementedError

        self.features = nn.Sequential(*self.features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channel, num_classes),
        )

        self.init_params()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, link.nn.SyncBatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def mobilenet_v3(**kwargs):
    """
    Constructs a MobileNet-V3 model.
    """
    model = MobileNetV3(**kwargs)
    return model
