import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer import WeightNet, WeightNet_DW
from prototype.spring.linklink.nn import SyncBatchNorm2d
from prototype.prototype.utils.misc import get_bn

BN = None

__all__ = ['shufflenet_v2_x0_5_weightnet', 'shufflenet_v2_x1_0_weightnet',
           'shufflenet_v2_x1_5_weightnet', 'shufflenet_v2_x2_0_weightnet']


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        # a reduction layer to reduce the input channel of grouped fc layer
        self.reduce = nn.Conv2d(inp, max(16, inp//16), 1, 1, 0, bias=True)

        self.wnet1 = WeightNet(inp, mid_channels, 1, 1)
        self.bn1 = BN(mid_channels)

        self.wnet2 = WeightNet_DW(mid_channels, ksize, stride)
        self.bn2 = BN(mid_channels)

        self.wnet3 = WeightNet(mid_channels, outputs, 1, 1)
        self.bn3 = BN(outputs)

        if stride == 2:  # down sample layer
            self.wnet_proj_1 = WeightNet_DW(inp, ksize, stride)
            self.bn_proj_1 = BN(inp)

            self.wnet_proj_2 = WeightNet(inp, inp, 1, 1)
            self.bn_proj_2 = BN(inp)

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x

        # global average pooling followed by channel reduction
        x_gap = x.mean(axis=2, keepdims=True).mean(axis=3, keepdims=True)
        x_gap = self.reduce(x_gap)

        x = self.wnet1(x, x_gap)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.wnet2(x, x_gap)
        x = self.bn2(x)

        x = self.wnet3(x, x_gap)
        x = self.bn3(x)
        x = F.relu(x)
        if self.stride == 2:
            x_proj = self.wnet_proj_1(x_proj, x_gap)
            x_proj = self.bn_proj_1(x_proj)
            x_proj = self.wnet_proj_2(x_proj, x_gap)
            x_proj = self.bn_proj_2(x_proj)
            x_proj = F.relu(x_proj)

        return torch.cat([x_proj, x], 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.shape
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.transpose(0, 1)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class ShuffleNetV2(nn.Module):
    def __init__(self, input_size=224, num_classes=1000,
                 model_size="1.5x", bn=None):
        super(ShuffleNetV2, self).__init__()

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        r"""The number of channels are slightly reduced to
            make WeightNet's FLOPs comparable to shufflenet baselines.
        """
        if model_size == "0.5x":
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif model_size == "1.0x":
            self.stage_out_channels = [-1, 24, 112, 224, 448, 1024]
        elif model_size == "1.5x":
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == "2.0x":
            self.stage_out_channels = [-1, 24, 248, 496, 992, 1024]
        else:
            raise NotImplementedError

        global BN

        BN = get_bn(bn)

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=True),
            BN(input_channel),
            nn.ReLU(),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                if i == 0:
                    self.features.append(
                        ShuffleV2Block(
                            input_channel, output_channel,
                            mid_channels=output_channel // 2,
                            ksize=3, stride=2,
                        )
                    )
                else:
                    self.features.append(
                        ShuffleV2Block(
                            input_channel // 2, output_channel,
                            mid_channels=output_channel // 2,
                            ksize=3, stride=1,
                        )
                    )

                input_channel = output_channel

        self.features = nn.Sequential(*self.features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, self.stage_out_channels[-1],
                      1, 1, 0, bias=True),
            BN(self.stage_out_channels[-1]),
            nn.ReLU(),
        )
        self.globalpool = nn.AvgPool2d(7)
        if self.model_size == "2.0x":
            self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(self.stage_out_channels[-1], num_classes, bias=True))
        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)

        x = self.globalpool(x)
        if self.model_size == "2.0x":
            x = self.dropout(x)
        x = x.reshape(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SyncBatchNorm2d)\
                    or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0/float(n))
                m.bias.data.zero_()


def shufflenet_v2_x2_0_weightnet(**kwargs):
    return ShuffleNetV2(model_size="2.0x", **kwargs)


def shufflenet_v2_x1_5_weightnet(**kwargs):
    return ShuffleNetV2(model_size="1.5x", **kwargs)


def shufflenet_v2_x1_0_weightnet(**kwargs):
    return ShuffleNetV2(model_size="1.0x", **kwargs)


def shufflenet_v2_x0_5_weightnet(**kwargs):
    return ShuffleNetV2(model_size="0.5x", **kwargs)
