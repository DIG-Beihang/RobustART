import torch.nn as nn
import math
from prototype.spring.linklink.nn import SyncBatchNorm2d
from prototype.prototype.utils.misc import get_logger, get_bn

BN = None

__all__ = ['preact_resnet18', 'preact_resnet34', 'preact_resnet50', 'preact_resnet101', 'preact_resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class PreactBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, preactivate=True):
        super(PreactBasicBlock, self).__init__()
        self.pre_bn = self.pre_relu = None
        if preactivate:
            self.pre_bn = BN(inplanes)
            self.pre_relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BN(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride
        self.preactivate = preactivate

    def forward(self, x):
        if self.preactivate:
            preact = self.pre_bn(x)
            preact = self.pre_relu(preact)
        else:
            preact = x

        out = self.conv1(preact)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(preact)
        else:
            residual = x

        out += residual

        return out


class PreactBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, preactivate=True):
        super(PreactBottleneck, self).__init__()
        self.pre_bn = self.pre_relu = None
        if preactivate:
            self.pre_bn = BN(inplanes)
            self.pre_relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BN(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BN(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.preactivate = preactivate

    def forward(self, x):
        if self.preactivate:
            preact = self.pre_bn(x)
            preact = self.pre_relu(preact)
        else:
            preact = x

        out = self.conv1(preact)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(preact)
        else:
            residual = x

        out += residual

        return out


class PreactResNet(nn.Module):
    """PreactResNet model class, based on
    `"Identity Mappings in Deep Residual Networks" <https://arxiv.org/abs/1603.05027>`_
    """
    def __init__(self, block, layers, num_classes=1000, deep_stem=False,
                 avg_down=False, bypass_last_bn=False,
                 bn=None):
        r"""
        Arguments:

        - layers (:obj:`list` of 4 ints): how many layers in each stage
        - num_classes (:obj:`int`): number of classification classes
        - deep_stem (:obj:`bool`): whether to use deep_stem as the first conv
        - avg_down (:obj:`bool`): whether to use avg_down when spatial downsample
        - bypass_last_bn (:obj:`bool`): whether use bypass_last_bn
        - bn (:obj:`dict`): definition of batchnorm
        """

        super(PreactResNet, self).__init__()

        logger = get_logger(__name__)

        global BN, bypass_bn_weight_list

        BN = get_bn(bn)
        bypass_bn_weight_list = []

        self.inplanes = 64
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.logger = get_logger(__name__)

        if self.deep_stem:
            self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                        BN(32),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                        BN(32),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BN(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.final_bn = BN(512 * block.expansion)
        self.final_relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif (isinstance(m, SyncBatchNorm2d) or isinstance(m, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0/float(n))
                m.bias.data.zero_()

        if bypass_last_bn:
            for param in bypass_bn_weight_list:
                param.data.zero_()
            logger.info('bypass {} bn.weight in BottleneckBlocks'.format(len(bypass_bn_weight_list)))

    def _make_layer(self, block, planes, blocks, stride=1, avg_down=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(stride, stride=stride, ceil_mode=True, count_include_pad=False),
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=1, bias=False),
                    # BN(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    # BN(planes * block.expansion),
                )

        # On the first residual block in the first residual layer we don't pre-activate,
        # because we take care of that (+ maxpool) after the initial conv layer
        preactivate_first = stride != 1

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, preactivate_first))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.final_bn(x)
        x = self.final_relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def preact_resnet18(**kwargs):
    """
    Constructs a Preact-ResNet-18 model.
    """
    model = PreactResNet(PreactBasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def preact_resnet34(**kwargs):
    """
    Constructs a Preact-ResNet-34 model.
    """
    model = PreactResNet(PreactBasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def preact_resnet50(**kwargs):
    """
    Constructs a Preact-ResNet-50 model.
    """
    model = PreactResNet(PreactBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def preact_resnet101(**kwargs):
    """
    Constructs a Preact-ResNet-101 model.
    """
    model = PreactResNet(PreactBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def preact_resnet152(**kwargs):
    """
    Constructs a Preact-ResNet-152 model.
    """
    model = PreactResNet(PreactBottleneck, [3, 8, 36, 3], **kwargs)
    return model
