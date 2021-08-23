import torch.nn as nn
import torch
import numpy as np
from prototype.prototype.utils.misc import get_bn


BN = None


__all__ = ['repvgg_A0', 'repvgg_A1', 'repvgg_A2',
           'repvgg_B0', 'repvgg_B1', 'repvgg_B1g2',
           'repvgg_B1g4', 'repvgg_B2', 'repvgg_B2g2',
           'repvgg_B2g4', 'repvgg_B3', 'repvgg_B3g2', 'repvgg_B3g4']


def conv_bn(in_planes, out_planes, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=padding, groups=groups, bias=False))
    result.add_module('bn', BN(out_planes))
    return result


class RepVGGBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_planes = in_planes

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups,
                                         bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = BN(in_planes) if out_planes == in_planes and stride == 1 else None
            self.rbr_dense = conv_bn(in_planes=in_planes, out_planes=out_planes,
                                     kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_planes=in_planes, out_planes=out_planes,
                                   kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_planes // self.groups
                kernel_value = np.zeros((self.in_planes, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_planes):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                kernel = torch.from_numpy(kernel_value).to(branch.weight.device)
            else:
                kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel.detach().cpu().numpy(), bias.detach().cpu().numpy(),


class RepVGG(nn.Module):

    def __init__(self,
                 num_blocks,
                 num_classes=1000,
                 width_multiplier=None,
                 override_groups_map=None,
                 bn=None,
                 deploy=False):

        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        global BN

        BN = get_bn(bn)

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_planes=3, out_planes=self.in_planes,
                                  kernel_size=3, stride=2, padding=1,
                                  deploy=self.deploy)
        self.cur_layer_idx = 1
        self.stage1 = self._make_layer(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_layer(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_layer(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_layer(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_planes=self.in_planes, out_planes=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l2: 2 for l2 in optional_groupwise_layers}
g4_map = {l4: 4 for l4 in optional_groupwise_layers}


def repvgg_A0(**kwargs):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, **kwargs)


def repvgg_A1(**kwargs):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, **kwargs)


def repvgg_A2(**kwargs):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, **kwargs)


def repvgg_B0(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, **kwargs)


def repvgg_B1(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None, **kwargs)


def repvgg_B1g2(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, **kwargs)


def repvgg_B1g4(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, **kwargs)


def repvgg_B2(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, **kwargs)


def repvgg_B2g2(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, **kwargs)


def repvgg_B2g4(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, **kwargs)


def repvgg_B3(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=None, **kwargs)


def repvgg_B3g2(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, **kwargs)


def repvgg_B3g4(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, **kwargs)
