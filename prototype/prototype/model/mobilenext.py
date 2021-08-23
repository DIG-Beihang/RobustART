import math
import torch
import torch.nn as nn
from torch.nn import init

import prototype.spring.linklink as link
from prototype.prototype.utils.misc import get_bn

__all__ = ['mobilenext']


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


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                      padding, groups=groups, bias=False),
            BN(out_planes),
            nn.ReLU6(inplace=True)
        )


class SandGlass(nn.Module):
    """Rethinking Bottleneck Structure for Efficient Mobile Network Design.
    depthwise -> linear pointwise -> pointwise -> depthwise
    """
    def __init__(self, inp, oup, stride, expand_ratio, identity_tensor_multiplier=1.0):
        super(SandGlass, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = inp // expand_ratio
        if hidden_dim < oup / 6.:
            hidden_dim = math.ceil(oup / 6.)
            hidden_dim = _make_divisible(hidden_dim, 16)

        # hidden_dim = int(round(inp // expand_ratio))
        self.use_identity = False if identity_tensor_multiplier == 1.0 else True
        self.identity_tensor_channels = int(round(inp * identity_tensor_multiplier))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        # dw
        layers.append(ConvBNReLU(inp, inp, kernel_size=3, stride=1, groups=inp))
        if expand_ratio != 1:
            # pw-linear
            layers.extend([
                nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                BN(hidden_dim),
            ])
        layers.extend([
            # pw
            ConvBNReLU(hidden_dim, oup, kernel_size=1, stride=1, groups=1),
            # dw-linear
            nn.Conv2d(oup, oup, kernel_size=3, stride=stride, groups=oup, padding=1, bias=False),
            BN(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            if self.use_identity:
                # reducing multiplier
                identity_tensor = x[:, :self.identity_tensor_channels, :, :] \
                    + out[:, :self.identity_tensor_channels, :, :]
                out = torch.cat([identity_tensor, out[:, self.identity_tensor_channels:, :, :]], dim=1)
                return out
            else:
                return x + out
        else:
            return out


class MobileNeXt(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 scale=1.0,
                 identity_tensor_multiplier=1.0,
                 sand_glass_setting=None,
                 round_nearest=8,
                 block=None,
                 dropout=0.0,
                 bn=None):
        """
        MobileNeXt main class
        Args:
            num_classes (int): Number of classes
            scale (float): Width multiplier - adjusts number of channels in each layer by this amount
            identity_tensor_multiplier(float): Identity tensor multiplier - reduce the number
            of element-wise additions in each block
            sand_glass_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            bn: Module specifying the normalization layer to use
        """
        super(MobileNeXt, self).__init__()

        global BN
        BN = get_bn(bn)

        if block is None:
            block = SandGlass
        input_channel = 32
        last_channel = 1280

        # building first layer
        input_channel = _make_divisible(input_channel * scale, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, scale), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]

        if sand_glass_setting is None:
            sand_glass_setting = [
                # t, c,  b, s
                [2, 96,  1, 2],
                [6, 144, 1, 1],
                [6, 192, 3, 2],
                [6, 288, 3, 2],
                [6, 384, 4, 1],
                [6, 576, 4, 2],
                [6, 960, 2, 1],
                [6, self.last_channel / scale, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(sand_glass_setting) == 0 or len(sand_glass_setting[0]) != 4:
            raise ValueError("sand_glass_setting should be non-empty "
                             "or a 4-element list, got {}".format(sand_glass_setting))

        # building sand glass blocks
        for t, c, b, s in sand_glass_setting:
            output_channel = _make_divisible(c * scale, round_nearest)
            for i in range(b):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t,
                          identity_tensor_multiplier=identity_tensor_multiplier))
                input_channel = output_channel

        # building last several layers
        # features.append(ConvBNReLU(nput_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, link.nn.SyncBatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    init.constant_(m.weight, 1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def mobilenext(**kwargs):
    """
    Constructs a MobileNeXt model.
    """
    model = MobileNeXt(**kwargs)
    return model
