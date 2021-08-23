import torch
from torch import nn
import prototype.spring.linklink as link
from prototype.prototype.utils.misc import get_logger, get_bn
from .layer import CondConv2d, BasicRouter


__all__ = ['mobilenetv2_condconv_pointwise',
           'mobilenetv2_condconv_independent',
           'mobilenetv2_condconv_shared']


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


class CondConvBNReLU(nn.Module):
    r"""Independent routing weights.
    """

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1,
                 num_experts=1, combine_kernel=False):
        padding = (kernel_size - 1) // 2
        super(CondConvBNReLU, self).__init__()
        self.condconv = CondConv2d(in_planes, out_planes, kernel_size, stride, padding,
                                   groups=groups, bias=False,
                                   num_experts=num_experts, combine_kernel=combine_kernel)
        self.bn = BN(out_planes)
        self.relu = nn.ReLU6(inplace=True)

        self.router = BasicRouter(in_planes, num_experts)

    def forward(self, x):
        routing_weight = self.router(x)
        x = self.condconv(x, routing_weight)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CondConvBNReLUV2(nn.Module):
    r"""Shared routing weights.
    """

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1,
                 num_experts=1, combine_kernel=False):
        padding = (kernel_size - 1) // 2
        super(CondConvBNReLUV2, self).__init__()
        self.condconv = CondConv2d(in_planes, out_planes, kernel_size, stride, padding,
                                   groups=groups, bias=False,
                                   num_experts=num_experts, combine_kernel=combine_kernel)
        self.bn = BN(out_planes)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x, routing_weight):
        x = self.condconv(x, routing_weight)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, num_experts, combine_kernel):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim,
                       stride=stride, groups=hidden_dim),
        ])
        self.conv = nn.Sequential(*layers)
        # pw-linear using condconv
        self.condconv = CondConv2d(
            hidden_dim, oup, 1, 1, 0, bias=False,
            num_experts=num_experts, combine_kernel=combine_kernel
        )
        self.bn = BN(oup)

        self.router = BasicRouter(inp, num_experts)

    def forward(self, x):
        routing_weight = self.router(x)
        output = self.conv(x)
        output = self.condconv(output, routing_weight)
        output = self.bn(output)
        if self.use_res_connect:
            return x + output
        else:
            return output


class CondConvInvertedResidual(nn.Module):
    r"""Replace all layers with CondConv.
    And the routing weights are independent.
    """

    def __init__(self, inp, oup, stride, expand_ratio, num_experts, combine_kernel):
        super(CondConvInvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw using condconv
            layers.append(
                CondConvBNReLU(inp, hidden_dim, kernel_size=1,
                               num_experts=num_experts,
                               combine_kernel=combine_kernel))
        layers.extend([
            # dw using condconv
            CondConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim,
                           num_experts=num_experts,
                           combine_kernel=combine_kernel),
        ])
        self.conv = nn.Sequential(*layers)
        # pw-linear using condconv
        self.condconv = CondConv2d(
            hidden_dim, oup, 1, 1, 0, bias=False, num_experts=num_experts,
            combine_kernel=combine_kernel)
        self.bn = BN(oup)

        self.router = BasicRouter(inp, num_experts)

    def forward(self, x):
        routing_weight = self.router(x)
        output = self.conv(x)
        output = self.condconv(output, routing_weight)
        output = self.bn(output)
        if self.use_res_connect:
            return x + output
        else:
            return output


class CondConvInvertedResidualV2(nn.Module):
    r"""Replace all layers with CondConv.
    And the routing weights are shared.
    """

    def __init__(self, inp, oup, stride, expand_ratio, num_experts, combine_kernel):
        super(CondConvInvertedResidualV2, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        self.expand_ratio = expand_ratio

        # pw using condconv
        if expand_ratio != 1:
            self.pw = CondConvBNReLUV2(
                inp, hidden_dim, kernel_size=1, num_experts=num_experts,
                combine_kernel=combine_kernel)
        # dw using condconv
        self.dw = CondConvBNReLUV2(
            hidden_dim, hidden_dim, stride=stride, groups=hidden_dim,
            num_experts=num_experts, combine_kernel=combine_kernel)
        # pw-linear using condconv
        self.pw_linear = CondConv2d(
            hidden_dim, oup, 1, 1, 0, bias=False, num_experts=num_experts,
            combine_kernel=combine_kernel)
        self.bn = BN(oup)

        self.router = BasicRouter(inp, num_experts)

    def forward(self, x):
        routing_weight = self.router(x)
        if self.expand_ratio != 1:
            output = self.pw(x, routing_weight)
            output = self.dw(output, routing_weight)
        else:
            output = self.dw(x, routing_weight)
        output = self.pw_linear(output, routing_weight)
        output = self.bn(output)
        if self.use_res_connect:
            return x + output
        else:
            return output


class MobileNetV2CondConv(nn.Module):
    r"""MobileNetV2CondConv main class

    Args:
        num_classes (int): Number of classes
        scale (float): Width multiplier - adjusts number of channels in each layer by this amount
        inverted_residual_setting: Network structure
        round_nearest (int): Round the number of channels in each layer to be a multiple of this number
        Set to 1 to turn off rounding
        block: Module specifying inverted residual building block for mobilenet
        num_experts (int): Number of experts for mixture. Default: 1
        final_condconv (bool): If ``True``, replace the finalconv with condconv
        fc_conv (bool): If ``True``, replace the fc with condconv
        combine_kernel (bool):
            If ``True``, first combine kernels, then use the combined kernel for the forward;
            If ``False``, first forward with different kernels, then combine the transformed features.
            Default: False

    """

    def __init__(self,
                 num_classes=1000,
                 scale=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=InvertedResidual,
                 dropout=0.2,
                 bn=None,
                 num_experts=1,
                 final_condconv=False,
                 fc_condconv=False,
                 combine_kernel=False):

        super(MobileNetV2CondConv, self).__init__()

        global BN
        BN = get_bn(bn)
        self.logger = get_logger(__name__)

        self.fc_condconv = fc_condconv
        self.logger.info(
            'Number of experts is {}'.format(num_experts))
        self.logger.info(
            'Replace finalconv with CondConv: {}'.format(final_condconv))
        self.logger.info(
            'Replace fc with CondConv: {}'.format(fc_condconv))
        self.logger.info(
            'Combine kernels to implement CondConv: {}'.format(combine_kernel))

        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * scale, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, scale), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * scale, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t,
                          num_experts=num_experts, combine_kernel=combine_kernel)
                )
                input_channel = output_channel
        # building last several layers
        if final_condconv:
            features.append(CondConvBNReLU(
                input_channel, self.last_channel, kernel_size=1,
                num_experts=num_experts, combine_kernel=combine_kernel))
        else:
            features.append(ConvBNReLU(
                input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # building classifier
        if fc_condconv:
            # change kernel_size to the size of feature maps
            self.dropout = nn.Dropout(0.2)
            self.classifier = CondConv2d(
                self.last_channel, num_classes, kernel_size=1, bias=False,
                num_experts=num_experts, combine_kernel=combine_kernel
            )
            self.classifier_router = BasicRouter(
                self.last_channel, num_experts
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.last_channel, num_classes),
            )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, link.nn.SyncBatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        if self.fc_condconv:
            x = self.avgpool(x)
            x = self.dropout(x)
            routing_weight = self.classifier_router(x)
            x = self.classifier(x, routing_weight)
            x = x.squeeze_()
        else:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def mobilenetv2_condconv_pointwise(**kwargs):
    r"""Replace the last pointwise in the InvertedResidual with CondConv.
    """
    kwargs['block'] = InvertedResidual
    model = MobileNetV2CondConv(**kwargs)
    return model


def mobilenetv2_condconv_independent(**kwargs):
    r"""Replace all convolutional layers in the InvertedResidual with CondConv.
    And the routing weights are independent.
    """
    kwargs['block'] = CondConvInvertedResidual
    model = MobileNetV2CondConv(**kwargs)
    return model


def mobilenetv2_condconv_shared(**kwargs):
    r"""Replace all convolutional layers in the InvertedResidual with CondConv.
    And the routing weights are shared.
    """
    kwargs['block'] = CondConvInvertedResidualV2
    model = MobileNetV2CondConv(**kwargs)
    return model
