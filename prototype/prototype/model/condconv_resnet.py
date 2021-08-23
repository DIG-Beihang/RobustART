import torch
import torch.nn as nn

from prototype.spring.linklink.nn import SyncBatchNorm2d
from prototype.prototype.utils.misc import get_logger, get_bn
from .layer import CondConv2d, BasicRouter


__all__ = ['resnet18_condconv_shared', 'resnet18_condconv_independent',
           'resnet34_condconv_shared', 'resnet34_condconv_independent',
           'resnet50_condconv_shared', 'resnet50_condconv_independent',
           'resnet101_condconv_shared', 'resnet101_condconv_independent',
           'resnet152_condconv_shared', 'resnet152_condconv_independent']


def drop_expert(x, training=False, drop_prob=0.):
    r"""Drop experts randomly during training

    Args:
        training (bool):
            If ``True``, during training, conduct random dropping
            If ``False``, during inference, do not conduct random dropping
        drop_prob (float): drop rate for experts. Default: 0.0
    """
    if training and drop_prob > 0.:
        keep_prob = 1.0 - drop_prob
        random_tensor = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        random_tensor = random_tensor + keep_prob
        binary_mask = torch.floor(random_tensor)
        x = (x / keep_prob) * binary_mask
    return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlockShared(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 num_experts=1, drop_prob=0., combine_kernel=False):
        super(BasicBlockShared, self).__init__()
        if norm_layer is None:
            norm_layer = BN
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlockShared only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlockShared")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = CondConv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
            num_experts=num_experts, combine_kernel=combine_kernel
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = CondConv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
            num_experts=num_experts, combine_kernel=combine_kernel
        )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.drop_prob = drop_prob

        self.router = BasicRouter(inplanes, num_experts)

    def forward(self, x):
        routing_weight = self.router(x)
        routing_weight = drop_expert(
            routing_weight, self.training, self.drop_prob)

        identity = x

        out = self.conv1(x, routing_weight)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, routing_weight)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlockIndependent(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 num_experts=1, drop_prob=0., combine_kernel=False):
        super(BasicBlockIndependent, self).__init__()
        if norm_layer is None:
            norm_layer = BN
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlockIndependent only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlockIndependent")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = CondConv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
            num_experts=num_experts, combine_kernel=combine_kernel
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = CondConv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
            num_experts=num_experts, combine_kernel=combine_kernel
        )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.drop_prob = drop_prob

        self.router1 = BasicRouter(inplanes, num_experts)
        self.router2 = BasicRouter(planes, num_experts)

    def forward(self, x):
        routing_weight1 = self.router1(x)
        routing_weight1 = drop_expert(
            routing_weight1, self.training, self.drop_prob)

        identity = x

        out = self.conv1(x, routing_weight1)
        out = self.bn1(out)
        out = self.relu(out)

        routing_weight2 = self.router2(out)
        routing_weight2 = drop_expert(
            routing_weight2, self.training, self.drop_prob)

        out = self.conv2(out, routing_weight2)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckShared(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 num_experts=1, drop_prob=0., combine_kernel=False):
        super(BottleneckShared, self).__init__()
        if norm_layer is None:
            norm_layer = BN
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = CondConv2d(
            inplanes, width, kernel_size=1, stride=1, padding=0, bias=False, num_experts=num_experts,
            combine_kernel=combine_kernel
        )
        self.bn1 = norm_layer(width)
        self.conv2 = CondConv2d(
            width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False,
            dilation=dilation, num_experts=num_experts, combine_kernel=combine_kernel
        )
        self.bn2 = norm_layer(width)
        self.conv3 = CondConv2d(
            width, planes * self.expansion, kernel_size=1, stride=1, padding=0,
            bias=False, num_experts=num_experts, combine_kernel=combine_kernel
        )
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.drop_prob = drop_prob

        self.router = BasicRouter(inplanes, num_experts)

    def forward(self, x):
        routing_weight = self.router(x)
        routing_weight = drop_expert(
            routing_weight, self.training, self.drop_prob)

        identity = x

        out = self.conv1(x, routing_weight)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, routing_weight)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out, routing_weight)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckIndependent(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 num_experts=1, drop_prob=0., combine_kernel=False):
        super(BottleneckIndependent, self).__init__()
        if norm_layer is None:
            norm_layer = BN
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = CondConv2d(
            inplanes, width, kernel_size=1, stride=1, padding=0, bias=False,
            num_experts=num_experts, combine_kernel=combine_kernel
        )
        self.bn1 = norm_layer(width)
        self.conv2 = CondConv2d(
            width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False,
            dilation=dilation, num_experts=num_experts, combine_kernel=combine_kernel
        )
        self.bn2 = norm_layer(width)
        self.conv3 = CondConv2d(
            width, planes * self.expansion, kernel_size=1, stride=1, padding=0,
            bias=False, num_experts=num_experts, combine_kernel=combine_kernel
        )
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.drop_prob = drop_prob

        self.router1 = BasicRouter(inplanes, num_experts)
        self.router2 = BasicRouter(width, num_experts)
        self.router3 = BasicRouter(width, num_experts)

    def forward(self, x):
        routing_weight1 = self.router1(x)
        routing_weight1 = drop_expert(
            routing_weight1, self.training, self.drop_prob)

        identity = x

        out = self.conv1(x, routing_weight1)
        out = self.bn1(out)
        out = self.relu(out)

        routing_weight2 = self.router2(out)
        routing_weight2 = drop_expert(
            routing_weight2, self.training, self.drop_prob)

        out = self.conv2(out, routing_weight2)
        out = self.bn2(out)
        out = self.relu(out)

        routing_weight3 = self.router3(out)
        routing_weight3 = drop_expert(
            routing_weight3, self.training, self.drop_prob)

        out = self.conv3(out, routing_weight3)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetCondConv(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None,
                 deep_stem=False,
                 avg_down=False,
                 bn=None,
                 dropout=0.,
                 num_experts=1,
                 drop_prob=0.,
                 combine_kernel=False):

        super(ResNetCondConv, self).__init__()

        global BN
        self.logger = get_logger(__name__)

        BN = get_bn(bn)
        if norm_layer is None:
            norm_layer = BN

        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.num_experts = num_experts
        self.logger.info(
            'Number of experts is {}'.format(num_experts))
        self.drop_prob = drop_prob
        self.logger.info(
            'DropExpert Rate: {}'.format(self.drop_prob))
        self.combine_kernel = combine_kernel
        self.logger.info(
            'Combine kernels to implement CondConv: {}'.format(combine_kernel))

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        if self.deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2,
                          padding=1, bias=False),
                norm_layer(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1,
                          padding=1, bias=False),
                norm_layer(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=1,
                          padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                                   stride=2, padding=3, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, SyncBatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckShared) or isinstance(m, BottleneckIndependent):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BottleneckShared) or isinstance(m, BottleneckIndependent):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(stride, stride=stride,
                                 ceil_mode=True, count_include_pad=False),
                    conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, self.num_experts,
                            self.drop_prob, self.combine_kernel))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, num_experts=self.num_experts,
                                drop_prob=self.drop_prob, combine_kernel=self.combine_kernel))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def resnet18_condconv_shared(**kwargs):
    model = ResNetCondConv(BasicBlockShared, [2, 2, 2, 2], **kwargs)
    return model


def resnet18_condconv_independent(**kwargs):
    model = ResNetCondConv(BasicBlockIndependent, [2, 2, 2, 2], **kwargs)
    return model


def resnet34_condconv_shared(**kwargs):
    model = ResNetCondConv(BasicBlockShared, [3, 4, 6, 3], **kwargs)
    return model


def resnet34_condconv_independent(**kwargs):
    model = ResNetCondConv(BasicBlockIndependent, [3, 4, 6, 3], **kwargs)
    return model


def resnet50_condconv_shared(**kwargs):
    model = ResNetCondConv(BottleneckShared, [3, 4, 6, 3], **kwargs)
    return model


def resnet50_condconv_independent(**kwargs):
    model = ResNetCondConv(BottleneckIndependent, [3, 4, 6, 3], **kwargs)
    return model


def resnet101_condconv_shared(**kwargs):
    model = ResNetCondConv(BottleneckShared, [3, 4, 23, 3], **kwargs)
    return model


def resnet101_condconv_independent(**kwargs):
    model = ResNetCondConv(BottleneckIndependent, [3, 4, 23, 3], **kwargs)
    return model


def resnet152_condconv_shared(**kwargs):
    model = ResNetCondConv(BottleneckShared, [3, 8, 36, 3], **kwargs)
    return model


def resnet152_condconv_independent(**kwargs):
    model = ResNetCondConv(BottleneckIndependent, [3, 8, 36, 3], **kwargs)
    return model
