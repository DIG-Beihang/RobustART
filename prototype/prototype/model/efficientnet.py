import torch.nn as nn
import torch
import math
from prototype.spring.linklink.nn import SyncBatchNorm2d
import torch.nn.functional as F
from torch.nn import init

import re
import collections
from collections import OrderedDict

from prototype.prototype.utils.misc import get_logger, get_bn

BN = None

__all__ = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
           'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
           'efficientnet_b0_nodrop', 'efficientnet_b1_nodrop', 'efficientnet_b2_nodrop', 'efficientnet_b3_nodrop',
           'efficientnet_b4_nodrop', 'efficientnet_b5_nodrop', 'efficientnet_b6_nodrop', 'efficientnet_b7_nodrop']


GlobalParams = collections.namedtuple('GlobalParams', [
    'dropout_rate', 'data_format', 'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate',
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def efficientnet_params(model_name):
    """Get efficientnet params based on model name."""
    params_dict = {
        # (width_coefficient, depth_coefficient, resolution, dropout_rate)
        'efficientnet_b0': (1.0, 1.0, 224, 0.2),
        'efficientnet_b1': (1.0, 1.1, 240, 0.2),
        'efficientnet_b2': (1.1, 1.2, 260, 0.3),
        'efficientnet_b3': (1.2, 1.4, 300, 0.3),
        'efficientnet_b4': (1.4, 1.8, 380, 0.4),
        'efficientnet_b5': (1.6, 2.2, 456, 0.4),
        'efficientnet_b6': (1.8, 2.6, 528, 0.5),
        'efficientnet_b7': (2.0, 3.1, 600, 0.5),
        'efficientnet_b0_nodrop': (1.0, 1.0, 224, 0),
        'efficientnet_b1_nodrop': (1.0, 1.1, 240, 0),
        'efficientnet_b2_nodrop': (1.1, 1.2, 260, 0),
        'efficientnet_b3_nodrop': (1.2, 1.4, 300, 0),
        'efficientnet_b4_nodrop': (1.4, 1.8, 380, 0),
        'efficientnet_b5_nodrop': (1.6, 2.2, 456, 0),
        'efficientnet_b6_nodrop': (1.8, 2.6, 528, 0),
        'efficientnet_b7_nodrop': (2.0, 3.1, 600, 0),
    }
    return params_dict[model_name]


def efficientnet(width_coefficient=None, depth_coefficient=None,
                 dropout_rate=0.2, drop_connect_rate=0.3, override_block=None):
    """Creates a efficientnet model."""
    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    if override_block is not None:
        assert isinstance(override_block, dict)
        for k, v in override_block.items():
            blocks_args[int(k)] = v
        logger = get_logger(__name__)
        logger.info('overrided blocks_args: {}'.format(blocks_args))
    global_params = GlobalParams(dropout_rate=dropout_rate,
                                 drop_connect_rate=drop_connect_rate,
                                 data_format='channels_last',
                                 num_classes=1000,
                                 width_coefficient=width_coefficient,
                                 depth_coefficient=depth_coefficient,
                                 depth_divisor=8,
                                 min_depth=None)
    decoder = BlockDecoder()
    return decoder.decode(blocks_args), global_params


class BlockDecoder(object):
    """Block Decoder for readability."""

    def _decode_block_string(self, block_string):
        """Gets a block through a string notation of arguments."""
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            strides=[int(options['s'][0]), int(options['s'][1])])

    def _encode_block_string(self, block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if block.se_ratio > 0 and block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    def decode(self, string_list):
        """Decodes a list of string notations to specify blocks inside the network.
        Args:
            string_list: a list of strings, each string is a notation of block.
        Returns:
            A list of namedtuples to represent blocks arguments.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(self._decode_block_string(block_string))
        return blocks_args

    def encode(self, blocks_args):
        """Encodes a list of Blocks to a list of strings.
        Args:
            blocks_args: A list of namedtuples to represent blocks arguments.
        Returns:
            a list of strings, each string is a notation of block.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(self._encode_block_string(block))
        return block_strings


def get_model_params(model_name, override_params=None, override_block=None):
    """Get the block args and global params for a given model."""
    if model_name.startswith('efficientnet'):
        width_coefficient, depth_coefficient, _, dropout_rate = (efficientnet_params(model_name))
        blocks_args, global_params = efficientnet(width_coefficient, depth_coefficient,
                                                  dropout_rate, override_block=override_block)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)

    if override_params is not None:
        # ValueError will be raised here if override_params has fields not included
        # in global_params.
        global_params = global_params._replace(**override_params)

    logger = get_logger(__name__)
    logger.info(blocks_args)
    logger.info(global_params)

    return blocks_args, global_params


def round_filters(filters, global_params):
    """Round number of filters based on depth multiplier."""
    orig_f = filters
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    logger = get_logger(__name__)
    logger.info('round_filter input={} output={}'.format(orig_f, new_filters))
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(x, training=False, drop_connect_rate=None):
    if drop_connect_rate is None:
        raise RuntimeError("drop_connect_rate not given")
    if not training:
        return x
    else:
        keep_prob = 1.0 - drop_connect_rate

        n = x.size(0)
        random_tensor = torch.rand([n, 1, 1, 1], dtype=x.dtype, device=x.device)
        random_tensor = random_tensor + keep_prob
        binary_mask = torch.floor(random_tensor)

        x = (x / keep_prob) * binary_mask

        return x


class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


def activation(act_type='swish'):
    if act_type == 'swish':
        act = swish()
        return act
    else:
        act = nn.ReLU(inplace=True)
        return act


class MBConvBlock(nn.Module):
    def __init__(self, block_args):
        super(MBConvBlock, self).__init__()

        self._block_args = block_args

        self.has_se = (self._block_args.se_ratio is not None) and \
            (self._block_args.se_ratio > 0) and \
            (self._block_args.se_ratio <= 1)

        self._build(inp=self._block_args.input_filters, oup=self._block_args.output_filters,
                    expand_ratio=self._block_args.expand_ratio, kernel_size=self._block_args.kernel_size,
                    stride=self._block_args.strides)

    def block_args(self):
        return self._block_args

    def _build(self, inp, oup, expand_ratio, kernel_size, stride):
        module_lists = []

        self.use_res_connect = all([s == 1 for s in stride]) and inp == oup

        if expand_ratio != 1:
            module_lists.append(nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False))
            module_lists.append(BN(inp * expand_ratio))
            module_lists.append(activation())

        module_lists.append(nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel_size,
                                      stride, kernel_size // 2, groups=inp * expand_ratio, bias=False))
        module_lists.append(BN(inp * expand_ratio))
        module_lists.append(activation())

        self.in_conv = nn.Sequential(*module_lists)

        if self.has_se:
            se_size = max(1, int(inp * self._block_args.se_ratio))
            s = OrderedDict()
            s['conv1'] = nn.Conv2d(inp * expand_ratio, se_size, kernel_size=1, stride=1, padding=0)
            s['act1'] = activation()
            s['conv2'] = nn.Conv2d(se_size, inp * expand_ratio, kernel_size=1, stride=1, padding=0)
            s['act2'] = nn.Sigmoid()
            self.se_block = nn.Sequential(s)

        self.out_conv = nn.Sequential(
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            BN(oup)
        )

    def forward(self, x, drop_connect_rate=None):
        out = self.in_conv(x)
        if self.has_se:
            weight = F.adaptive_avg_pool2d(out, output_size=1)
            weight = self.se_block(weight)
            out = out * weight

        out = self.out_conv(out)
        if self._block_args.id_skip:
            if self.use_res_connect:
                if drop_connect_rate is not None:
                    out = drop_connect(out, self.training, drop_connect_rate)
                out = out + x

        return out


class EfficientNet(nn.Module):
    """EfficientNet class, based on
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_
    """
    def __init__(self,
                 blocks_args=None,
                 global_params=None,
                 use_fc_bn=False,
                 fc_bn_init_scale=1.0,
                 bn=None):
        super(EfficientNet, self).__init__()

        global BN

        BN = get_bn(bn)

        if not isinstance(blocks_args, list):
            raise ValueError('blocks_args should be a list.')

        self.logger = get_logger(__name__)

        self._global_params = global_params
        self._blocks_args = blocks_args
        self.use_fc_bn = use_fc_bn
        self.fc_bn_init_scale = fc_bn_init_scale

        self._build()

    def _build(self):
        blocks = []
        for block_args in self._blocks_args:
            assert block_args.num_repeat > 0

            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            blocks.append(MBConvBlock(block_args))

            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, strides=[1, 1])

            for _ in range(block_args.num_repeat - 1):
                blocks.append(MBConvBlock(block_args))
        self.blocks = nn.ModuleList(blocks)

        c_in = round_filters(32, self._global_params)
        self.stem = nn.Sequential(
            nn.Conv2d(3, c_in, kernel_size=3, stride=2, padding=1, bias=False),
            BN(c_in),
            activation(),
        )
        c_in = round_filters(320, self._global_params)
        c_final = round_filters(1280, self._global_params)
        self.head = nn.Sequential(
            nn.Conv2d(c_in, c_final, kernel_size=1, stride=1, padding=0, bias=False),
            BN(c_final),
            activation(),
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = torch.nn.Linear(c_final, self._global_params.num_classes)

        if self._global_params.dropout_rate > 0:
            self.dropout = nn.Dropout2d(p=self._global_params.dropout_rate, inplace=True)
        else:
            self.dropout = None

        self._initialize_weights()

        if self.use_fc_bn:
            self.logger.info('using fc_bn, init scale={}'.format(self.fc_bn_init_scale))
            self.fc_bn = BN(self._global_params.num_classes)
            init.constant_(self.fc_bn.weight, self.fc_bn_init_scale)
            init.constant_(self.fc_bn.bias, 0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, SyncBatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0/float(n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.stem(x)

        for idx in range(len(self.blocks)):
            drop_rate = self._global_params.drop_connect_rate
            if drop_rate:
                drop_rate *= float(idx) / len(self.blocks)
            x = self.blocks[idx](x, drop_rate)
        x = self.head(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)

        if self.use_fc_bn and x.size(0) > 1:
            x = self.fc_bn(x.view(x.size(0), -1, 1, 1))
            x = x.view(x.size(0), -1)

        return x


def efficientnet_b0(override_params=None, override_block=None, **kwargs):
    """
    Constructs a EfficientNet-B0 model.
    """
    model_name = 'efficientnet_b0'
    blocks_args, global_params = get_model_params(model_name, override_params, override_block)

    model = EfficientNet(blocks_args, global_params, **kwargs)

    return model


def efficientnet_b1(override_params=None, **kwargs):
    """
    Constructs a EfficientNet-B1 model.
    """
    model_name = 'efficientnet_b1'
    blocks_args, global_params = get_model_params(model_name, override_params)

    model = EfficientNet(blocks_args, global_params, **kwargs)

    return model


def efficientnet_b2(override_params=None, **kwargs):
    """
    Constructs a EfficientNet-B2 model.
    """
    model_name = 'efficientnet_b2'
    blocks_args, global_params = get_model_params(model_name, override_params)

    model = EfficientNet(blocks_args, global_params, **kwargs)

    return model


def efficientnet_b3(override_params=None, **kwargs):
    """
    Constructs a EfficientNet-B3 model.
    """
    model_name = 'efficientnet_b3'
    blocks_args, global_params = get_model_params(model_name, override_params)

    model = EfficientNet(blocks_args, global_params, **kwargs)

    return model


def efficientnet_b4(override_params=None, **kwargs):
    """
    Constructs a EfficientNet-B4 model.
    """
    model_name = 'efficientnet_b4'
    blocks_args, global_params = get_model_params(model_name, override_params)

    model = EfficientNet(blocks_args, global_params, **kwargs)

    return model


def efficientnet_b5(override_params=None, **kwargs):
    """
    Constructs a EfficientNet-B5 model.
    """
    model_name = 'efficientnet_b5'
    blocks_args, global_params = get_model_params(model_name, override_params)

    model = EfficientNet(blocks_args, global_params, **kwargs)

    return model


def efficientnet_b6(override_params=None, **kwargs):
    """
    Constructs a EfficientNet-B6 model.
    """
    model_name = 'efficientnet_b6'
    blocks_args, global_params = get_model_params(model_name, override_params)

    model = EfficientNet(blocks_args, global_params, **kwargs)

    return model


def efficientnet_b7(override_params=None, **kwargs):
    """
    Constructs a EfficientNet-B7 model.
    """
    model_name = 'efficientnet_b7'
    blocks_args, global_params = get_model_params(model_name, override_params)

    model = EfficientNet(blocks_args, global_params, **kwargs)

    return model


def efficientnet_b0_nodrop(override_params=None, override_block=None, **kwargs):
    """
    Constructs a EfficientNet-B0 model.
    """
    model_name = 'efficientnet_b0_nodrop'
    blocks_args, global_params = get_model_params(model_name, override_params, override_block)

    model = EfficientNet(blocks_args, global_params, **kwargs)

    return model


def efficientnet_b1_nodrop(override_params=None, **kwargs):
    """
    Constructs a EfficientNet-B1 model.
    """
    model_name = 'efficientnet_b1_nodrop'
    blocks_args, global_params = get_model_params(model_name, override_params)

    model = EfficientNet(blocks_args, global_params, **kwargs)

    return model


def efficientnet_b2_nodrop(override_params=None, **kwargs):
    """
    Constructs a EfficientNet-B2 model.
    """
    model_name = 'efficientnet_b2_nodrop'
    blocks_args, global_params = get_model_params(model_name, override_params)

    model = EfficientNet(blocks_args, global_params, **kwargs)

    return model


def efficientnet_b3_nodrop(override_params=None, **kwargs):
    """
    Constructs a EfficientNet-B3 model.
    """
    model_name = 'efficientnet_b3_nodrop'
    blocks_args, global_params = get_model_params(model_name, override_params)

    model = EfficientNet(blocks_args, global_params, **kwargs)

    return model


def efficientnet_b4_nodrop(override_params=None, **kwargs):
    """
    Constructs a EfficientNet-B4 model.
    """
    model_name = 'efficientnet_b4_nodrop'
    blocks_args, global_params = get_model_params(model_name, override_params)

    model = EfficientNet(blocks_args, global_params, **kwargs)

    return model


def efficientnet_b5_nodrop(override_params=None, **kwargs):
    """
    Constructs a EfficientNet-B5 model.
    """
    model_name = 'efficientnet_b5_nodrop'
    blocks_args, global_params = get_model_params(model_name, override_params)

    model = EfficientNet(blocks_args, global_params, **kwargs)

    return model


def efficientnet_b6_nodrop(override_params=None, **kwargs):
    """
    Constructs a EfficientNet-B6 model.
    """
    model_name = 'efficientnet_b6_nodrop'
    blocks_args, global_params = get_model_params(model_name, override_params)

    model = EfficientNet(blocks_args, global_params, **kwargs)

    return model


def efficientnet_b7_nodrop(override_params=None, **kwargs):
    """
    Constructs a EfficientNet-B7 model.
    """
    model_name = 'efficientnet_b7_nodrop'
    blocks_args, global_params = get_model_params(model_name, override_params)

    model = EfficientNet(blocks_args, global_params, **kwargs)

    return model