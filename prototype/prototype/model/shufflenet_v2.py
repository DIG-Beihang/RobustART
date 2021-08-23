import torch
import torch.nn as nn
import math
from prototype.spring.linklink.nn import SyncBatchNorm2d
from prototype.prototype.utils.misc import get_bn

BN = None

__all__ = ['shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
           'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'shufflenet_v2_scale']


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3,
                                    stride=self.stride, padding=1),
                BN(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1,
                          stride=1, padding=0, bias=False),
                BN(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            BN(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features,
                                kernel_size=3, stride=self.stride, padding=1),
            BN(branch_features),
            nn.Conv2d(branch_features, branch_features,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BN(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    """ShuffleNet model class, based on
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" <https://arxiv.org/abs/1807.11164>`_
    """
    def __init__(self, stages_repeats, stages_out_channels, num_classes=1000, bn=None):
        r"""
        - stages_repeats (:obj:`list` of 3 ints): how many layers in each stage
        - stages_out_channels (:obj:`list` of 5 ints): output channels
        - num_classes (:obj:`int`): number of classification classes
        - bn (:obj:`dict`): definition of batchnorm
        """
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError(
                'expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError(
                'expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        global BN

        BN = get_bn(bn)

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            BN(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(
                    output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            BN(output_channels),
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(output_channels, num_classes)

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def shufflenet_v2_x0_5(**kwargs):
    """
    Constructs a ShuffleNet-V2-0.5 model.
    """
    model = ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)
    return model


def shufflenet_v2_x1_0(**kwargs):
    """
    Constructs a ShuffleNet-V2-1.0 model.
    """
    model = ShuffleNetV2([4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)
    return model


def shufflenet_v2_x1_5(**kwargs):
    """
    Constructs a ShuffleNet-V2-1.5 model.
    """
    model = ShuffleNetV2([4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)
    return model


def shufflenet_v2_x2_0(**kwargs):
    """
    Constructs a ShuffleNet-V2-2.0 model.
    """
    model = ShuffleNetV2([4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)
    return model


def shufflenet_v2_scale(**kwargs):
    """
    Constructs a custom ShuffleNet-V2 model.
    """
    model = ShuffleNetV2(**kwargs)
    return model
