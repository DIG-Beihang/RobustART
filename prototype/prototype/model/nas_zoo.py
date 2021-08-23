import torch
import torch.nn as nn

from functools import partial
from prototype.prototype.utils.misc import get_bn
from prototype.spring.linklink.nn import SyncBatchNorm2d

BN = None


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


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.time = 0
        self.netpara = 0

    def forward(self, input, *args):
        return input


class Rec(nn.Module):
    def __init__(self,
                 inplanes,
                 outplanes,
                 size_in,
                 stride=1,
                 k=3,
                 t=6,
                 forward=True):
        super(Rec, self).__init__()

        padding = k // 2
        self.time = 0
        self.size_in = size_in
        self.size_out = (size_in - 1 + stride) // stride

        if forward:
            self.conv1 = nn.Conv2d(inplanes,
                                   inplanes * t,
                                   kernel_size=1,
                                   bias=False)
            self.bn1 = BN(inplanes * t)

            self.conv2_1 = nn.Conv2d(inplanes * t,
                                     inplanes * t,
                                     kernel_size=(1, k),
                                     stride=(1, stride),
                                     padding=(0, padding),
                                     bias=False)
            self.bn2_1 = BN(inplanes * t)
            self.conv2_2 = nn.Conv2d(inplanes * t,
                                     inplanes * t,
                                     kernel_size=(k, 1),
                                     stride=(stride, 1),
                                     padding=(padding, 0),
                                     bias=False)
            self.bn2_2 = BN(inplanes * t)

            self.conv3 = nn.Conv2d(inplanes * t,
                                   outplanes,
                                   kernel_size=1,
                                   bias=False)
            self.bn3 = BN(outplanes)

            self.activation = nn.ReLU(inplace=True)

        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

        self.netpara = sum(p.numel() for p in self.parameters()) / 1e6

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2_1(out)
        out = self.bn2_1(out)
        out = self.activation(out)

        out = self.conv2_2(out)
        out = self.bn2_2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out


class DualConv(nn.Module):
    def __init__(self,
                 inplanes,
                 outplanes,
                 size_in,
                 stride=1,
                 k=3,
                 t=6,
                 forward=True):
        super(DualConv, self).__init__()
        padding = k // 2
        self.time = 0
        self.size_in = size_in
        self.size_out = (size_in - 1 + stride) // stride

        if forward:
            self.conv1 = nn.Conv2d(inplanes,
                                   inplanes * t,
                                   kernel_size=1,
                                   bias=False)
            self.bn1 = BN(inplanes * t)

            self.conv2_1 = nn.Conv2d(inplanes * t,
                                     inplanes * t,
                                     kernel_size=k,
                                     stride=1,
                                     padding=padding,
                                     bias=False)
            self.bn2_1 = BN(inplanes * t)
            self.conv2_2 = nn.Conv2d(inplanes * t,
                                     inplanes * t,
                                     kernel_size=k,
                                     stride=stride,
                                     padding=padding,
                                     bias=False)
            self.bn2_2 = BN(inplanes * t)

            self.conv3 = nn.Conv2d(inplanes * t,
                                   outplanes,
                                   kernel_size=1,
                                   bias=False)
            self.bn3 = BN(outplanes)
            self.activation = nn.ReLU(inplace=True)

        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

        self.netpara = sum(p.numel() for p in self.parameters()) / 1e6

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2_1(out)
        out = self.bn2_1(out)
        out = self.activation(out)

        out = self.conv2_2(out)
        out = self.bn2_2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out


class NormalConv(nn.Module):
    def __init__(self,
                 inplanes,
                 outplanes,
                 size_in,
                 stride=1,
                 k=3,
                 t=6,
                 forward=True):
        super(NormalConv, self).__init__()
        padding = k // 2
        self.time = 0
        self.size_in = size_in
        self.size_out = (size_in - 1 + stride) // stride

        if forward:
            self.conv1 = nn.Conv2d(inplanes,
                                   inplanes * t,
                                   kernel_size=1,
                                   bias=False)
            self.bn1 = BN(inplanes * t)

            self.conv2 = nn.Conv2d(inplanes * t,
                                   inplanes * t,
                                   kernel_size=k,
                                   stride=stride,
                                   padding=padding,
                                   bias=False)
            self.bn2 = BN(inplanes * t)

            self.conv3 = nn.Conv2d(inplanes * t,
                                   outplanes,
                                   kernel_size=1,
                                   bias=False)
            self.bn3 = BN(outplanes)
            self.activation = nn.ReLU(inplace=True)

        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

        self.netpara = sum(p.numel() for p in self.parameters()) / 1e6

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out


class LinearBottleneck(nn.Module):
    def __init__(self,
                 inplanes,
                 outplanes,
                 size_in,
                 stride=1,
                 k=3,
                 t=6,
                 activation=nn.ReLU,
                 forward=True,
                 group=1,
                 dilation=1):
        super(LinearBottleneck, self).__init__()
        dk = k + (dilation - 1) * 2
        padding = dk // 2
        self.time = 0
        self.size_in = size_in
        self.size_out = (size_in - 1 + stride) // stride

        if forward:
            self.conv1 = nn.Conv2d(inplanes,
                                   inplanes * t,
                                   kernel_size=1,
                                   bias=False,
                                   groups=group)
            self.bn1 = BN(inplanes * t)
            self.conv2 = nn.Conv2d(inplanes * t,
                                   inplanes * t,
                                   kernel_size=k,
                                   stride=stride,
                                   padding=padding,
                                   bias=False,
                                   groups=inplanes * t,
                                   dilation=dilation)
            self.bn2 = BN(inplanes * t)
            self.conv3 = nn.Conv2d(inplanes * t,
                                   outplanes,
                                   kernel_size=1,
                                   bias=False,
                                   groups=group)
            self.bn3 = BN(outplanes)
            self.activation = activation(inplace=True)

        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

        self.netpara = sum(p.numel() for p in self.parameters()) / 1e6

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out


def channel_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]


class ShuffleBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 outplanes,
                 size_in,
                 stride=1,
                 k=3,
                 activation=nn.ReLU,
                 forward=True,
                 group=1):
        super(ShuffleBlock, self).__init__()
        padding = k // 2
        self.time = 0
        self.size_in = size_in
        self.size_out = (size_in - 1 + stride) // stride

        if forward:
            self.conv1 = nn.Conv2d(inplanes,
                                   outplanes // 2,
                                   kernel_size=1,
                                   bias=False,
                                   groups=1)
            self.bn1 = BN(outplanes // 2)
            self.conv2 = nn.Conv2d(outplanes // 2,
                                   outplanes // 2,
                                   kernel_size=k,
                                   stride=stride,
                                   padding=padding,
                                   bias=False,
                                   groups=outplanes // 2)
            self.bn2 = BN(outplanes // 2)
            self.conv3 = nn.Conv2d(outplanes // 2,
                                   outplanes,
                                   kernel_size=1,
                                   bias=False,
                                   groups=1)
            self.bn3 = BN(outplanes)
            self.activation = activation(inplace=True)
        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inplanes,
                          inplanes,
                          k,
                          stride,
                          padding,
                          groups=inplanes,
                          bias=False),
                BN(inplanes),
                # pw-linear
                nn.Conv2d(inplanes, inplanes, 1, 1, 0, bias=False),
                BN(inplanes),
                activation(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)

        self.stride = stride
        self.inplanes = inplanes
        self.outplanes = outplanes

        self.netpara = sum(p.numel() for p in self.parameters()) / 1e6

    def forward(self, x):
        if self.stride == 1:
            x_proj, x = channel_shuffle(x)
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.activation(out)

            out = self.conv2(out)
            out = self.bn2(out)

            out = self.conv3(out)
            out = self.bn3(out)
            out = self.activation(out)
            return torch.cat((x_proj, self.branch_main(out)), 1)
        elif self.stride == 2:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.activation(out)

            out = self.conv2(out)
            out = self.bn2(out)

            out = self.conv3(out)
            out = self.bn3(out)
            out = self.activation(out)
            return torch.cat((self.branch_proj(x), self.branch_main(out)), 1)
        else:
            raise


class FBNetValCell(nn.Module):
    candidate_num = 8
    candidates = []
    for k in [3, 5]:
        for t in [1, 3, 6]:
            candidates.append(partial(LinearBottleneck, k=k, t=t))
            if t == 1:
                candidates.append(partial(LinearBottleneck, k=k, t=t, group=2))

    def __init__(self, cin, size_in, stride, cout, branch):
        super(FBNetValCell, self).__init__()
        self.stride = stride
        self.cin = cin
        self.cout = cout
        self.size_in = size_in

        self.candidates = nn.ModuleList()

        if self.cin == self.cout and self.stride == 1 and \
                branch == SupValCell.candidate_num:
            self.path = Identity()
        else:
            self.path = FBNetValCell.candidates[branch](inplanes=self.cin,
                                                        outplanes=self.cout,
                                                        size_in=self.size_in,
                                                        stride=self.stride)

    def forward(self, curr_layer):
        """Runs the conv cell."""
        if self.cin == self.cout and self.stride == 1:
            return self.path(curr_layer) + curr_layer
        else:
            return self.path(curr_layer)


class SupValCell(nn.Module):
    candidate_num = 24
    candidates = []
    for k in [3, 5, 7]:
        for t in [1, 3, 6]:
            candidates.append(partial(LinearBottleneck, k=k, t=t))
    for t in [1, 2, 4]:
        candidates.append(partial(NormalConv, k=3, t=t))
    for t in [1, 2, 4]:
        candidates.append(partial(DualConv, k=3, t=t))
    for k in [3, 5, 7]:
        for t in [1, 2, 4]:
            candidates.append(partial(Rec, k=k, t=t))

    def __init__(self, cin, size_in, stride, cout, branch, keep_prob=-1):
        super(SupValCell, self).__init__()
        self.stride = stride
        self.cin = cin
        self.cout = cout
        self.size_in = size_in

        if self.cin == self.cout and self.stride == 1 and \
                branch == SupValCell.candidate_num:
            self.path = Identity()
        else:
            self.path = SupValCell.candidates[branch](inplanes=self.cin,
                                                      outplanes=self.cout,
                                                      size_in=self.size_in,
                                                      stride=self.stride)

    def forward(self, curr_layer):
        """Runs the conv cell."""
        if self.cin == self.cout and self.stride == 1:
            return self.path(curr_layer) + curr_layer
        else:
            return self.path(curr_layer)


class AlignedSupValCell(nn.Module):
    candidate_num = 19
    candidates = []
    for k in [3, 5, 7]:
        for t in [1, 3, 6]:
            candidates.append(partial(LinearBottleneck, k=k, t=t))
    for t in [1, 2]:
        candidates.append(partial(NormalConv, k=3, t=t))
    for t in [1, 2]:
        candidates.append(partial(DualConv, k=3, t=t))
    for k in [5, 7]:
        for t in [1, 2, 4]:
            candidates.append(partial(Rec, k=k, t=t))

    def __init__(self, cin, size_in, stride, cout, branch, keep_prob=-1):
        super(AlignedSupValCell, self).__init__()
        self.stride = stride
        self.cin = cin
        self.cout = cout
        self.size_in = size_in

        if self.cin == self.cout and self.stride == 1 and \
                branch == AlignedSupValCell.candidate_num:
            self.path = Identity()
        else:
            self.path = \
                AlignedSupValCell.candidates[branch](inplanes=self.cin,
                                                     outplanes=self.cout,
                                                     size_in=self.size_in,
                                                     stride=self.stride)

    def forward(self, curr_layer):
        """Runs the conv cell."""
        if self.cin == self.cout and self.stride == 1:
            return self.path(curr_layer) + curr_layer
        else:
            return self.path(curr_layer)


class MBValCell(nn.Module):
    candidate_num = 19
    candidates = []
    for k in [3, 5, 7]:
        for t in [1, 3, 6]:
            candidates.append(partial(LinearBottleneck, k=k, t=t))
    for t in [1, 2]:
        candidates.append(partial(NormalConv, k=3, t=t))
    for t in [1, 2]:
        candidates.append(partial(DualConv, k=3, t=t))
    for k in [5, 7]:
        for t in [1, 2, 4]:
            candidates.append(partial(Rec, k=k, t=t))

    def __init__(self, cin, size_in, stride, cout, branches, keep_prob=-1):
        super(MBValCell, self).__init__()
        self.stride = stride
        self.cin = cin
        self.cout = cout
        self.size_in = size_in
        self.pathes = nn.ModuleList()
        for branch in branches:
            self.pathes.append(MBValCell.candidates[branch](
                inplanes=self.cin,
                outplanes=self.cout,
                size_in=self.size_in,
                stride=self.stride))

    def forward(self, curr_layer):
        """Runs the conv cell."""
        if self.cin == self.cout and self.stride == 1:
            out = [path(curr_layer) for path in self.pathes]
            out.append(curr_layer)
            return sum(out) / len(out)
        else:
            out = [path(curr_layer) for path in self.pathes]
            return sum(out) / len(out)


class MACMBValCell(nn.Module):
    candidate_num = 27
    candidates = []

    for k in [3, 5, 7, 9, 11]:
        for t in [1, 3, 6]:
            if k == 3:
                for d in [1, 2, 3]:
                    candidates.append(
                        partial(LinearBottleneck, k=k, t=t, dilation=d))
            else:
                candidates.append(partial(LinearBottleneck, k=k, t=t))

    for k in [5, 7]:
        for t in [1, 2, 4]:
            candidates.append(partial(Rec, k=k, t=t))

    def __init__(self, cin, size_in, stride, cout, branches, keep_prob=-1):
        super(MACMBValCell, self).__init__()
        self.stride = stride
        self.cin = cin
        self.cout = cout
        self.size_in = size_in
        self.pathes = nn.ModuleList()
        for branch in branches:
            self.pathes.append(MACMBValCell.candidates[branch](
                inplanes=self.cin,
                outplanes=self.cout,
                size_in=self.size_in,
                stride=self.stride))

    def forward(self, curr_layer):
        """Runs the conv cell."""
        if self.cin == self.cout and self.stride == 1:
            out = [path(curr_layer) for path in self.pathes]
            out.append(curr_layer)
            return sum(out) / len(out)
        else:
            out = [path(curr_layer) for path in self.pathes]
            return sum(out) / len(out)


class MACMBLiteValCell(nn.Module):
    candidate_num = 20
    candidates = []

    for k in [3, 5, 7, 9, 11]:
        for t in [1, 2, 4, 8]:
            candidates.append(partial(LinearBottleneck, k=k, t=t))

    def __init__(self, cin, size_in, stride, cout, branches):
        super(MACMBLiteValCell, self).__init__()
        self.stride = stride
        self.cin = cin
        self.cout = cout
        self.size_in = size_in
        self.pathes = nn.ModuleList()
        for branch in branches:
            self.pathes.append(MACMBLiteValCell.candidates[branch](
                inplanes=self.cin,
                outplanes=self.cout,
                size_in=self.size_in,
                stride=self.stride))

    def forward(self, curr_layer):
        """Runs the conv cell."""
        if self.cin == self.cout and self.stride == 1:
            out = [path(curr_layer) for path in self.pathes]
            out.append(curr_layer)
            return sum(out) / len(out)
        else:
            out = [path(curr_layer) for path in self.pathes]
            return sum(out) / len(out)


class ValNet(nn.Module):
    def __init__(self,
                 scale,
                 channel_dist,
                 num_classes,
                 input_size,
                 Cell,
                 cell_seq,
                 use_aux,
                 keep_prob=-1,
                 bn=None):
        super(ValNet, self).__init__()
        global BN, bypass_bn_weight_list
        BN = get_bn(bn)

        self.use_aux = use_aux

        self._time = 0
        self.scale = scale
        self.c = list(channel_dist[:1]) + [
            _make_divisible(ch * self.scale, 8) for ch in channel_dist[1:]
        ]
        self.num_classes = num_classes
        self.input_size = input_size
        cell_seq = list(cell_seq)
        self.total_blocks = len(cell_seq)

        self._set_stem()

        self.arch_parameters = []
        self.cells = nn.ModuleList()

        self.stage = 0
        for cell_idx, (c, branch) in enumerate(cell_seq):
            if c == 'N':
                stride = 1

                cout = cin = self.c[self.stage]
            elif c == 'E':
                self.stage += 1
                stride = 1

                cout = self.c[self.stage]
                cin = self.c[self.stage - 1]
            elif c == 'R':
                self.stage += 1
                stride = 2
                cout = self.c[self.stage]
                cin = self.c[self.stage - 1]

                self.aux_idx = cell_idx + 1
            else:
                raise NotImplementedError(
                    'unimplemented cell type: {}'.format(c))
            curr_drop_path_keep_prob = \
                self.calculate_curr_drop_path_keep_prob(cell_idx, keep_prob)

            self.cells.append(
                Cell(cin,
                     self.curr_size,
                     stride,
                     cout,
                     branch,
                     keep_prob=curr_drop_path_keep_prob))
            self.curr_size = (self.curr_size - 1 + stride) // stride

        self._set_tail()

        self.netpara = sum(p.numel() for p in self.parameters()) / 1e6

        if self.use_aux:
            self._set_aux()

    def calculate_curr_drop_path_keep_prob(self, cell_idx,
                                           drop_path_keep_prob):
        layer_ratio = cell_idx / float(self.total_blocks)
        return 1 - layer_ratio * (1 - drop_path_keep_prob)

    def adjust_keep_prob(self, curr_epoch, epochs):
        ratio = float(curr_epoch) / epochs
        for cell_idx, cell in enumerate(self.cells):
            for path in cell.pathes:
                path.adjust_keep_prob(ratio)

    def _set_stem(self):
        raise NotImplementedError()

    def _set_tail(self):
        raise NotImplementedError()

    def _set_aux(self):
        raise NotImplementedError()

    def forward(self, x, arch_update=False):
        curr_layer = self.stem(x)
        for cell_idx, cell in enumerate(self.cells):
            curr_layer = cell(curr_layer)

            if cell_idx == self.aux_idx:
                x_aux = curr_layer

        curr_layer = self.last_conv(curr_layer)
        x = self.avgpool(curr_layer)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if self.use_aux:
            x_aux = self.aux(x_aux)
            x_aux = x_aux.view(x_aux.size(0), -1)
            x_aux = self.fc_aux(x_aux)
            return x, x_aux
        else:
            return x


class ValImageNet(ValNet):
    def __init__(self,
                 alloc_code,
                 scale=1.0,
                 channel_dist=(16, 32, 64, 128, 256),
                 num_classes=1000,
                 input_size=224,
                 alloc_space=(1, 4, 4, 8, 4),
                 cell_plan='super',
                 alloc_plan='NR',
                 use_aux=False,
                 bn=None):
        cell_seq = {
            'NR':
            lambda x: "N" * x[0] + "R" + "N" * x[1] + "R" + "N" * x[
                2] + "R" + "N" * x[3] + "R" + "N" * x[4],
            'NER':
            lambda x: "N" * x[0] + "R" + "N" * x[1] + "R" + "N" * x[2] + "R" +
            "N" * x[3] + "E" + "N" * x[4] + "R" + "N" * x[5] + "E",
        }[alloc_plan](alloc_space)
        cell_seq = zip(cell_seq, alloc_code)
        cell = {
            'super': SupValCell,
            'mb': MBValCell,
            'aligned': AlignedSupValCell,
            'macmb': MACMBValCell,
            'maclitemb': MACMBLiteValCell
        }[cell_plan]

        super(ValImageNet, self).__init__(scale,
                                          channel_dist,
                                          num_classes,
                                          input_size,
                                          cell,
                                          cell_seq,
                                          use_aux,
                                          bn=bn)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, SyncBatchNorm2d) or isinstance(
                    m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _set_stem(self):
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.c[0], 3, stride=2, padding=1, bias=False),
            BN(self.c[0]), nn.ReLU(inplace=True))
        self.curr_size = (self.input_size + 1) // 2

    def _set_tail(self):
        self.last_conv = nn.Sequential(
            nn.Conv2d(self.c[self.stage], 1024, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))

        self.avgpool = nn.AvgPool2d(self.curr_size)

        self.fc = nn.Linear(1024, self.num_classes)

    def _set_aux(self):
        self.aux = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(self.c[-1] if self.alloc_plan == "NR" else self.c[-2],
                      128, 1),  # hot fix
            BN(128),
            nn.ReLU(),
            nn.Conv2d(128, 768, (self.curr_size - 2) // 3),
            BN(768),
            nn.ReLU(),
        )
        self.fc_aux = nn.Linear(768, self.num_classes)


def mbnas_t29_x0_84(**kwargs):
    model = ValImageNet([[12], [10], [11], [11], [6, 11, 3], [11, 7, 10, 4],
                         [11, 9, 7, 1, 4], [11, 9, 16, 13, 7]],
                        scale=0.84,
                        channel_dist=[16, 32, 64, 128, 256],
                        alloc_space=[0, 0, 2, 0, 2],
                        cell_plan='mb',
                        alloc_plan='NR',
                        **kwargs)
    return model


def mbnas_t47_x1_00(**kwargs):
    model = ValImageNet(
        [[12, 17], [10, 6], [11], [11], [6, 11, 3], [3], [3],
         [12, 10, 7, 1, 4], [11, 10, 4, 2, 7], [9, 11, 16, 13, 4, 7]],
        scale=1.0,
        channel_dist=[16, 32, 64, 128, 256],
        alloc_space=[0, 0, 2, 2, 2],
        cell_plan='mb',
        alloc_plan='NR',
        **kwargs)
    return model


def supnas_t18_x1_00(**kwargs):
    model = ValImageNet([
        9, 17, 13, 19, 19, 19, 6, 0, 19, 19, 19, 3, 0, 3, 0, 0, 19, 19, 0, 19,
        4, 7, 1, 6, 6
    ],
        scale=1,
        channel_dist=[16, 32, 64, 128, 256],
        alloc_space=[1, 4, 4, 8, 4],
        cell_plan='aligned',
        alloc_plan='NR',
        **kwargs)
    return model


def supnas_t37_x0_92(**kwargs):
    model = ValImageNet([
        19, 15, 10, 19, 19, 1, 17, 13, 6, 0, 6, 11, 9, 0, 0, 3, 1, 6, 9, 9, 10,
        3, 5, 13, 7
    ],
        scale=0.92,
        channel_dist=[16, 32, 64, 128, 256],
        alloc_space=[1, 4, 4, 8, 4],
        cell_plan='aligned',
        alloc_plan='NR',
        **kwargs)
    return model


def supnas_t44_x1_00(**kwargs):
    model = ValImageNet([
        16, 17, 0, 9, 12, 11, 6, 1, 3, 11, 3, 4, 2, 3, 11, 3, 11, 3, 11, 6, 12,
        7, 7, 4, 6
    ],
        scale=1,
        channel_dist=[16, 32, 64, 128, 256],
        alloc_space=[1, 4, 4, 8, 4],
        cell_plan='aligned',
        alloc_plan='NR',
        **kwargs)
    return model


def supnas_t66_x1_11(**kwargs):
    model = ValImageNet([
        12, 15, 10, 11, 11, 11, 7, 6, 3, 11, 6, 5, 11, 11, 11, 6, 16, 11, 16,
        11, 2, 7, 11, 7, 7
    ],
        scale=1.11,
        channel_dist=[16, 32, 64, 128, 256],
        alloc_space=[1, 4, 4, 8, 4],
        cell_plan='aligned',
        alloc_plan='NR',
        **kwargs)
    return model


def supnas_t100_x0_96(**kwargs):
    model = ValImageNet([
        10, 15, 12, 3, 9, 9, 4, 4, 14, 17, 12, 8, 17, 5, 8, 13, 12, 5, 12, 13,
        5, 5, 8, 15, 1
    ],
        scale=0.96,
        channel_dist=[16, 32, 64, 128, 256],
        alloc_space=[1, 4, 4, 8, 4],
        cell_plan='aligned',
        alloc_plan='NR',
        **kwargs)
    return model


def nas_custom(**kwargs):
    assert 'alloc_code' in kwargs and 'cell_plan' in kwargs, 'Require alloc_code and cell_plan'
    alloc_code = kwargs['alloc_code']
    kwargs.pop('alloc_code')
    if kwargs['cell_plan'] == 'mobileV2':
        alloc_code = [(op if op != 9 else 19) for op in alloc_code]
    if kwargs['cell_plan'] in ['mobileV2', 'super']:
        cell_plan = 'aligned'
        kwargs.pop('cell_plan')
    model = ValImageNet(alloc_code=alloc_code, cell_plan=cell_plan, **kwargs)
    return model
