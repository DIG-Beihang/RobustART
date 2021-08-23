import torch.nn as nn
import math
from prototype.spring.linklink.nn import SyncBatchNorm2d
from prototype.prototype.utils.misc import get_bn

BN = None

__all__ = ['dmcp_resnet18_45M', 'dmcp_resnet18_47M',  'dmcp_resnet18_51M', 'dmcp_resnet18_1040M', 'dmcp_resnet50_282M',
           'dmcp_resnet50_1100M', 'dmcp_resnet50_2200M']

resnet18_47M = {
    'conv1': 8,
    'fc': 144,
    'layer1': {'0': {'conv1': [8, 8], 'conv2': [8, 8]}},
    'layer2': {'0': {'conv1': [8, 16], 'conv2': [16, 24]}},
    'layer3': {'0': {'conv1': [24, 40], 'conv2': [40, 48]}},
    'layer4': {'0': {'conv1': [48, 88], 'conv2': [88, 144]}, '1': {'conv1': [144, 96], 'conv2': [96, 144]}}
    }

resnet18_45M = {
    'conv1': 8,
    'fc': 136,
    'layer1': {'0': {'conv1': [8, 8], 'conv2': [8, 8]}},
    'layer2': {'0': {'conv1': [8, 16], 'conv2': [16, 24]}},
    'layer3': {'0': {'conv1': [24, 40], 'conv2': [40, 40]}, '1': {'conv1': [40, 32], 'conv2': [32, 40]}},
    'layer4': {'0': {'conv1': [40, 88], 'conv2': [88, 136]}, '1': {'conv1': [136, 88], 'conv2': [88, 136]}}
}

resnet18_51M = {
    'conv1': 8,
    'layer1': {'0': {'conv1': (8, 8), 'conv2': (8, 8)}, '1': {'conv1': (8, 8), 'conv2': (8, 8)}},
    'layer2': {'0': {'conv1': (8, 16), 'conv2': (16, 16)}, '1': {'conv1': (16, 16), 'conv2': (16, 16)}},
    'layer3': {'0': {'conv1': (16, 32), 'conv2': (32, 40)}, '1': {'conv1': (40, 24), 'conv2': (24, 40)}},
    'layer4': {'0': {'conv1': (40, 48), 'conv2': (48, 248)}, '1': {'conv1': (248, 40), 'conv2': (40, 248)}},
    'fc': 248,
}

resnet18_480M = {
    'conv1': 24,
    'layer1': {'0': {'conv1': (24, 16), 'conv2': (16, 24)}, '1': {'conv1': (24, 16), 'conv2': (16, 24)}},
    'layer2': {'0': {'conv1': (24, 40), 'conv2': (40, 72)}, '1': {'conv1': (72, 40), 'conv2': (40, 72)}},
    'layer3': {'0': {'conv1': (72, 120), 'conv2': (120, 192)}, '1': {'conv1': (192, 104), 'conv2': (104, 192)}},
    'layer4': {'0': {'conv1': (192, 264), 'conv2': (264, 496)}, '1': {'conv1': (496, 256), 'conv2': (256, 496)}},
    'fc': 496,
}

resnet18_1040M = {
    'conv1': 24,
    'layer1': {'0': {'conv1': (24, 16), 'conv2': (16, 24)}, '1': {'conv1': (24, 16), 'conv2': (16, 24)}},
    'layer2': {'0': {'conv1': (24, 40), 'conv2': (40, 72)}, '1': {'conv1': (72, 40), 'conv2': (40, 72)}},
    'layer3': {'0': {'conv1': (72, 120), 'conv2': (120, 192)}, '1': {'conv1': (192, 104), 'conv2': (104, 192)}},
    'layer4': {'0': {'conv1': (192, 264), 'conv2': (264, 496)}, '1': {'conv1': (496, 256), 'conv2': (256, 496)}},
    'fc': 496,
}

# 0.25x
resnet50_282M = {
    'conv1': 16,
    'layer1': {'0': {'conv1': [16, 8], 'conv2': [8, 8], 'conv3': [8, 32]},
               '1': {'conv1': [32, 8], 'conv2': [8, 8], 'conv3': [8, 32]},
               '2': {'conv1': [32, 8], 'conv2': [8, 8], 'conv3': [8, 32]}},
    'layer2': {'0': {'conv1': [32, 24], 'conv2': [24, 24], 'conv3': [24, 88]},
               '1': {'conv1': [88, 24], 'conv2': [24, 24], 'conv3': [24, 88]},
               '2': {'conv1': [88, 24], 'conv2': [24, 24], 'conv3': [24, 88]},
               '3': {'conv1': [88, 24], 'conv2': [24, 24], 'conv3': [24, 88]}},
    'layer3': {'0': {'conv1': [88, 64], 'conv2': [64, 72], 'conv3': [72, 248]},
               '1': {'conv1': [248, 32], 'conv2': [32, 56], 'conv3': [56, 248]},
               '2': {'conv1': [248, 40], 'conv2': [40, 64], 'conv3': [64, 248]},
               '3': {'conv1': [248, 48], 'conv2': [48, 72], 'conv3': [72, 248]},
               '4': {'conv1': [248, 56], 'conv2': [56, 80], 'conv3': [80, 248]},
               '5': {'conv1': [248, 64], 'conv2': [64, 64], 'conv3': [64, 248]}},
    'layer4': {'0': {'conv1': [248, 184], 'conv2': [184, 176], 'conv3': [176, 1304]},
               '1': {'conv1': [1304, 136], 'conv2': [136, 224], 'conv3': [224, 1304]},
               '2': {'conv1': [1304, 184], 'conv2': [184, 224], 'conv3': [224, 1304]}},
    'fc': 1304,
}

# 0.5x
resnet50_1100M = {
    'conv1': 48,
    'layer1': {'0': {'conv1': [48, 16], 'conv2': [16, 16], 'conv3': [16, 136]},
               '1': {'conv1': [136, 16], 'conv2': [16, 16], 'conv3': [16, 136]},
               '2': {'conv1': [136, 16], 'conv2': [16, 24], 'conv3': [24, 136]}},
    'layer2': {'0': {'conv1': [136, 24], 'conv2': [24, 56], 'conv3': [56, 288]},
               '1': {'conv1': [288, 40], 'conv2': [40, 48], 'conv3': [48, 288]},
               '2': {'conv1': [288, 32], 'conv2': [32, 40], 'conv3': [40, 288]},
               '3': {'conv1': [288, 40], 'conv2': [40, 56], 'conv3': [56, 288]}},
    'layer3': {'0': {'conv1': [288, 80], 'conv2': [80, 120], 'conv3': [120, 920]},
               '1': {'conv1': [920, 64], 'conv2': [64, 112], 'conv3': [112, 920]},
               '2': {'conv1': [920, 88], 'conv2': [88, 104], 'conv3': [104, 920]},
               '3': {'conv1': [920, 80], 'conv2': [80, 112], 'conv3': [112, 920]},
               '4': {'conv1': [920, 96], 'conv2': [96, 128], 'conv3': [128, 920]},
               '5': {'conv1': [920, 112], 'conv2': [112, 128], 'conv3': [128, 920]}},
    'layer4': {'0': {'conv1': [920, 256], 'conv2': [256, 392], 'conv3': [392, 1304]},
               '1': {'conv1': [1304, 312], 'conv2': [312, 392], 'conv3': [392, 1304]},
               '2': {'conv1': [1304, 400], 'conv2': [400, 288], 'conv3': [288, 1304]}},
    'fc': 1304,
}

# 0.75x
resnet50_2200M = {
    'conv1': 48,
    'layer1': {'0': {'conv1': [48, 16], 'conv2': [16, 24], 'conv3': [24, 120]},
               '1': {'conv1': [120, 16], 'conv2': [16, 24], 'conv3': [24, 120]},
               '2': {'conv1': [120, 24], 'conv2': [24, 24], 'conv3': [24, 120]}},
    'layer2': {'0': {'conv1': [120, 64], 'conv2': [64, 80], 'conv3': [80, 328]},
               '1': {'conv1': [328, 48], 'conv2': [48, 56], 'conv3': [56, 328]},
               '2': {'conv1': [328, 56], 'conv2': [56, 88], 'conv3': [88, 328]},
               '3': {'conv1': [328, 72], 'conv2': [72, 96], 'conv3': [96, 328]}},
    'layer3': {'0': {'conv1': [328, 200], 'conv2': [200, 224], 'conv3': [224, 936]},
               '1': {'conv1': [936, 96], 'conv2': [96, 184], 'conv3': [184, 936]},
               '2': {'conv1': [936, 176], 'conv2': [176, 216], 'conv3': [216, 936]},
               '3': {'conv1': [936, 200], 'conv2': [200, 224], 'conv3': [224, 936]},
               '4': {'conv1': [936, 216], 'conv2': [216, 232], 'conv3': [232, 936]},
               '5': {'conv1': [936, 224], 'conv2': [224, 232], 'conv3': [232, 936]}},
    'layer4': {'0': {'conv1': [936, 488], 'conv2': [488, 504], 'conv3': [504, 2048]},
               '1': {'conv1': [2048, 496], 'conv2': [496, 504], 'conv3': [504, 2048]},
               '2': {'conv1': [2048, 504], 'conv2': [504, 504], 'conv3': [504, 2048]}},
    'fc': 2048,
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, stride=1, downsample=None, bottleneck_settings=None):
        super(BasicBlock, self).__init__()
        conv1_in_ch, conv1_out_ch = bottleneck_settings['conv1']
        self.conv1 = conv3x3(conv1_in_ch, conv1_out_ch, stride)
        self.bn1 = BN(conv1_out_ch)
        self.relu = nn.ReLU(inplace=True)

        conv2_in_ch, conv2_out_ch = bottleneck_settings['conv2']
        self.conv2 = conv3x3(conv2_in_ch, conv2_out_ch)
        self.bn2 = BN(conv2_out_ch)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, stride=1, downsample=None, bottleneck_settings=None):
        super(Bottleneck, self).__init__()
        conv1_in_ch, conv1_out_ch = bottleneck_settings['conv1']
        self.conv1 = nn.Conv2d(conv1_in_ch, conv1_out_ch, kernel_size=1, bias=False)
        self.bn1 = BN(conv1_out_ch)

        conv2_in_ch, conv2_out_ch = bottleneck_settings['conv2']
        self.conv2 = nn.Conv2d(conv2_in_ch, conv2_out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BN(conv2_out_ch)

        conv3_in_ch, conv3_out_ch = bottleneck_settings['conv3']
        self.conv3 = nn.Conv2d(conv3_in_ch, conv3_out_ch, kernel_size=1, bias=False)
        self.bn3 = BN(conv3_out_ch)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Adaptive_ResNet(nn.Module):
    """Pruned Redidual Networks class, based on
    `"DMCP: Differentiable Markov Channel Pruning for Neural Networks" <https://arxiv.org/abs/2005.03354>`_
    """
    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 bn=None,
                 channel_config=None,
                 nnie_type=True):
        r"""
        Arguments:

        - block (:obj:`nn.Module`): block type
        - layers (:obj:`list` of 4 ints): how many layers in each stage
        - num_classes (:obj:`int`): number of classification classes
        - bn (:obj:`dict`): definition of batchnorm
        - channel_config (:obj:`dict`): configurations of the pruned channels
        - nnie_type (:obj:`bool`): if ``True``, the first maxpool is set with ceil_mode=True
        """

        super(Adaptive_ResNet, self).__init__()

        global BN

        BN = get_bn(bn)
        self.inplanes = 64
        conv1_out_ch = channel_config['conv1']
        self.conv1 = nn.Conv2d(3, conv1_out_ch, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BN(conv1_out_ch)
        self.relu = nn.ReLU(inplace=True)
        if nnie_type:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], bottleneck_settings=channel_config['layer1'])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, bottleneck_settings=channel_config['layer2'])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, bottleneck_settings=channel_config['layer3'])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, bottleneck_settings=channel_config['layer4'])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channel_config['fc'], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif (isinstance(m, SyncBatchNorm2d)
                  or isinstance(m, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, avg_down=False, bottleneck_settings=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            in_ch, _ = bottleneck_settings['0']['conv1']
            if 'conv3' in bottleneck_settings['0'].keys():
                _, out_ch = bottleneck_settings['0']['conv3']
            else:
                # basic block
                _, out_ch = bottleneck_settings['0']['conv2']
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch,
                          kernel_size=1, stride=stride, bias=False),
                BN(out_ch),
            )

        layers = []
        layers.append(block(stride, downsample, bottleneck_settings=bottleneck_settings['0']))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(bottleneck_settings=bottleneck_settings[str(i)]))

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

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def dmcp_resnet18_45M(**kwargs):
    """
    equal to ResNet18-1/8
    """
    kwargs['channel_config'] = resnet18_45M
    return Adaptive_ResNet(BasicBlock, [1, 1, 2, 2], **kwargs)


def dmcp_resnet18_47M(**kwargs):
    """
    equal to ResNet18-1/8
    """
    kwargs['channel_config'] = resnet18_47M
    return Adaptive_ResNet(BasicBlock, [1, 1, 1, 2], **kwargs)


def dmcp_resnet18_51M(**kwargs):
    """
    equal to ResNet18-1/8
    """
    kwargs['channel_config'] = resnet18_51M
    return Adaptive_ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def dmcp_resnet18_480M(**kwargs):
    """
    equal to ResNet18-1/2
    """
    kwargs['channel_config'] = resnet18_480M
    return Adaptive_ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def dmcp_resnet18_1040M(**kwargs):
    """
    equal to ResNet18-3/4
    """
    kwargs['channel_config'] = resnet18_1040M
    return Adaptive_ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def dmcp_resnet50_282M(**kwargs):
    """
    equal to ResNet50-1/4
    """
    kwargs['channel_config'] = resnet50_282M
    return Adaptive_ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def dmcp_resnet50_1100M(**kwargs):
    """
    equal to ResNet50-1/2
    """
    kwargs['channel_config'] = resnet50_1100M
    return Adaptive_ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def dmcp_resnet50_2200M(**kwargs):
    """
    equal to ResNet50-3/4
    """
    kwargs['channel_config'] = resnet50_2200M
    return Adaptive_ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
