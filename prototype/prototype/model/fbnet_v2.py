import copy
import logging
from collections import OrderedDict
import typing
import torch.nn as nn

from .layer import IRFBlock, IRPoolBlock, ConvBNRelu, Identity
from prototype.prototype.utils.fbnet_helper import merge, filter_kwargs, update_dict, unify_args, \
    get_merged_dict, add_dropout, get_divisible_by


__all__ = ['fbnetv2_f1', 'fbnetv2_f4', 'fbnetv2_l2_hs', 'fbnetv2_l3']

logger = logging.getLogger(__name__)


PRIMITIVES = {
    "skip": lambda in_channels, out_channels, stride, **kwargs: Identity(
        in_channels, out_channels, stride
    ),
    "conv": lambda in_channels, out_channels, stride, **kwargs: ConvBNRelu(
        in_channels,
        out_channels,
        **merge(conv_args={"stride": stride}, kwargs=kwargs)
    ),
    "conv_k1": lambda in_channels, out_channels, stride, **kwargs: ConvBNRelu(
        in_channels,
        out_channels,
        **merge(
            conv_args={"stride": stride, "kernel_size": 1, "padding": 0},
            kwargs=kwargs,
        )
    ),
    "conv_k3": lambda in_channels, out_channels, stride, **kwargs: ConvBNRelu(
        in_channels,
        out_channels,
        **merge(
            conv_args={"stride": stride, "kernel_size": 3, "padding": 1},
            kwargs=kwargs,
        )
    ),
    "conv_k5": lambda in_channels, out_channels, stride, **kwargs: ConvBNRelu(
        in_channels,
        out_channels,
        **merge(
            conv_args={"stride": stride, "kernel_size": 5, "padding": 2},
            kwargs=kwargs,
        )
    ),
    "conv_hs": lambda in_channels, out_channels, stride, **kwargs: ConvBNRelu(
        in_channels,
        out_channels,
        **merge(
            conv_args={"stride": stride}, relu_args="hswish", kwargs=kwargs
        )
    ),
    "conv_k1_hs": lambda in_channels, out_channels, stride, **kwargs: ConvBNRelu(
        in_channels,
        out_channels,
        **merge(
            conv_args={"stride": stride, "kernel_size": 1, "padding": 0},
            relu_args="hswish",
            kwargs=kwargs,
        )
    ),
    "conv_k3_hs": lambda in_channels, out_channels, stride, **kwargs: ConvBNRelu(
        in_channels,
        out_channels,
        **merge(
            conv_args={"stride": stride, "kernel_size": 3, "padding": 1},
            relu_args="hswish",
            kwargs=kwargs,
        )
    ),
    "conv_k5_hs": lambda in_channels, out_channels, stride, **kwargs: ConvBNRelu(
        in_channels,
        out_channels,
        **merge(
            conv_args={"stride": stride, "kernel_size": 5, "padding": 2},
            relu_args="hswish",
            kwargs=kwargs,
        )
    ),
    "irf": lambda in_channels, out_channels, stride, **kwargs: IRFBlock(
        in_channels, out_channels, stride=stride, **kwargs
    ),
    "ir_k3": lambda in_channels, out_channels, stride, **kwargs: IRFBlock(
        in_channels, out_channels, stride=stride, kernel_size=3, **kwargs
    ),
    "ir_k3_g2": lambda in_channels, out_channels, stride, **kwargs: IRFBlock(
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=3,
        pw_groups=2,
        **kwargs
    ),
    "ir_k5": lambda in_channels, out_channels, stride, **kwargs: IRFBlock(
        in_channels, out_channels, stride=stride, kernel_size=5, **kwargs
    ),
    "ir_k5_g2": lambda in_channels, out_channels, stride, **kwargs: IRFBlock(  # noqa
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=5,
        pw_groups=2,
        **kwargs
    ),
    "ir_k3_hs": lambda in_channels, out_channels, stride, **kwargs: IRFBlock(
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=3,
        relu_args="hswish",
        **kwargs
    ),
    "ir_k5_hs": lambda in_channels, out_channels, stride, **kwargs: IRFBlock(
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=5,
        relu_args="hswish",
        **kwargs
    ),
    "ir_k3_se": lambda in_channels, out_channels, stride, **kwargs: IRFBlock(
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=3,
        se_args="se",
        **kwargs
    ),
    "ir_k5_se": lambda in_channels, out_channels, stride, **kwargs: IRFBlock(
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=5,
        se_args="se",
        **kwargs
    ),
    "ir_k3_sehsig": lambda in_channels, out_channels, stride, **kwargs: IRFBlock(  # noqa
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=3,
        se_args="se_hsig",
        **kwargs
    ),
    "ir_k5_sehsig": lambda in_channels, out_channels, stride, **kwargs: IRFBlock(  # noqa
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=5,
        se_args="se_hsig",
        **kwargs
    ),
    "ir_k3_sehsig_hs": lambda in_channels, out_channels, stride, **kwargs: IRFBlock(  # noqa
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=3,
        relu_args="hswish",
        se_args="se_hsig",
        **kwargs
    ),
    "ir_k5_sehsig_hs": lambda in_channels, out_channels, stride, **kwargs: IRFBlock(  # noqa
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=5,
        relu_args="hswish",
        se_args="se_hsig",
        **kwargs
    ),
    "ir_pool": lambda in_channels, out_channels, stride, **kwargs: IRPoolBlock(  # noqa
        in_channels,
        out_channels,
        **filter_kwargs(IRPoolBlock, kwargs),
        stride=stride,
    ),
    "ir_pool_hs": lambda in_channels, out_channels, stride, **kwargs: IRPoolBlock(  # noqa
        in_channels,
        out_channels,
        **filter_kwargs(IRPoolBlock, kwargs),
        stride=stride,
        relu_args="hswish",
    ),
}


def get_i8f_models(model_def):
    ret = {}
    for name, arch in model_def.items():
        new_name = name + "_i8f"
        new_arch = copy.deepcopy(arch)
        if "basic_args" not in new_arch:
            new_arch["basic_args"] = {}
        new_arch["basic_args"]["dw_skip_bnrelu"] = True
        ret[new_name] = new_arch
    return ret


def _ex(x, always_pw=None):
    ret = {"expansion": x}
    if always_pw is not None:
        ret["always_pw"] = always_pw
    return ret


e6 = _ex(6)
e4 = _ex(4)
e3 = _ex(3)
e2 = _ex(2)
e1 = _ex(1)
e1p = _ex(1, always_pw=True)

BASIC_ARGS = {}

IRF_CFG = {"less_se_channels": False}

MODEL_ARCH_DMASKING_NET = {
    "dmasking_f1": {
        # nparams: 5.998952, nflops 55.747008
        "input_size": 128,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 8, 2, 1]],
            # stage 1
            [["ir_k5", 8, 1, 1, e1, IRF_CFG]],
            # stage 2
            [
                ["ir_k5", 24, 2, 1, _ex(5.4566), IRF_CFG],
                ["ir_k5", 24, 1, 1, _ex(4.7912), IRF_CFG],
            ],
            # stage 3
            [
                ["ir_k5_sehsig", 32, 2, 1, _ex(5.3501), IRF_CFG],
                ["ir_k5_sehsig", 24, 1, 1, _ex(4.5379), IRF_CFG],
            ],
            # stage 4
            [
                ["ir_k5_hs", 56, 2, 1, _ex(5.7133), IRF_CFG],
                ["ir_k3_hs", 56, 1, 1, _ex(4.1212), IRF_CFG],
                ["ir_k3_sehsig_hs", 56, 1, 1, _ex(5.1246), IRF_CFG],
                ["skip", 80, 1, 1, _ex(5.0333), IRF_CFG],
                ["ir_k5_sehsig_hs", 80, 1, 1, _ex(4.5070), IRF_CFG],
                ["ir_k5_sehsig_hs", 80, 1, 1, _ex(1.7712), IRF_CFG],
            ],
            # stage 5
            [
                ["ir_k3_sehsig_hs", 144, 2, 1, _ex(4.5685), IRF_CFG],
                ["ir_k5_sehsig_hs", 144, 1, 1, _ex(5.8400), IRF_CFG],
                ["ir_k5_sehsig_hs", 144, 1, 1, _ex(6.8754), IRF_CFG],
                ["skip", 224, 1, 1, _ex(6.5245), IRF_CFG],
            ],
            # stage 6
            [["ir_pool_hs", 1600, 1, 1, e6]],
        ],
    },
    "dmasking_f4": {
        # nparams: 6.993656, nflops 234.689136
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 16, 2, 1]],
            # stage 1
            [["ir_k3", 16, 1, 1, e1, IRF_CFG]],
            # stage 2
            [
                ["ir_k5", 24, 2, 1, _ex(5.4566), IRF_CFG],
                ["ir_k5", 24, 1, 1, _ex(1.7912), IRF_CFG],
                ["ir_k5", 24, 1, 1, _ex(1.7912), IRF_CFG],
            ],
            # stage 3
            [
                ["ir_k5_sehsig", 32, 2, 1, _ex(5.3501), IRF_CFG],
                ["ir_k5_sehsig", 32, 1, 1, _ex(3.5379), IRF_CFG],
                ["ir_k5_sehsig", 32, 1, 1, _ex(4.5379), IRF_CFG],
                ["ir_k5_sehsig", 32, 1, 1, _ex(4.5379), IRF_CFG],
            ],
            # stage 4
            [
                ["ir_k5_hs", 64, 2, 1, _ex(5.7133), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(2.1212), IRF_CFG],
                ["skip", 64, 1, 1, _ex(3.1246), IRF_CFG],
                ["ir_k3_hs", 104, 1, 1, _ex(5.0333), IRF_CFG],
                ["ir_k5_sehsig_hs", 104, 1, 1, _ex(2.5070), IRF_CFG],
                ["ir_k5_sehsig_hs", 104, 1, 1, _ex(1.7712), IRF_CFG],
                ["ir_k5_sehsig_hs", 112, 1, 1, _ex(3.7712), IRF_CFG],
            ],
            # stage 5
            [
                ["ir_k3_sehsig_hs", 184, 2, 1, _ex(5.5685), IRF_CFG],
                ["ir_k5_sehsig_hs", 184, 1, 1, _ex(2.8400), IRF_CFG],
                ["ir_k5_sehsig_hs", 184, 1, 1, _ex(4.8754), IRF_CFG],
                ["skip", 224, 1, 1, _ex(6.5245), IRF_CFG],
            ],
            # stage 6
            [["ir_pool_hs", 1984, 1, 1, e6]],
        ],
    },
    "dmasking_l2_hs": {
        # nparams: 8.49 nflops: 422.04
        "input_size": 256,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [["conv_k3_hs", 16, 2, 1]],
            [["ir_k3_hs", 16, 1, 1, e1, IRF_CFG]],
            [
                ["ir_k5_hs", 24, 2, 1, _ex(5.4566), IRF_CFG],
                ["ir_k5_hs", 24, 1, 1, _ex(1.7912), IRF_CFG],
                ["ir_k3_hs", 24, 1, 1, _ex(1.7912), IRF_CFG],
                ["ir_k5_hs", 24, 1, 1, _ex(1.7912), IRF_CFG],
            ],
            [
                ["ir_k5_sehsig", 40, 2, 1, _ex(5.3501), IRF_CFG],
                ["ir_k5_sehsig", 32, 1, 1, _ex(3.5379), IRF_CFG],
                ["ir_k5_sehsig", 32, 1, 1, _ex(4.5379), IRF_CFG],
                ["ir_k5_sehsig", 32, 1, 1, _ex(4.5379), IRF_CFG],
            ],
            [
                ["ir_k5_hs", 64, 2, 1, _ex(5.7133), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(2.1212), IRF_CFG],
                ["skip", 64, 1, 1, _ex(3.1246), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(3.1246), IRF_CFG],
                ["ir_k3_hs", 112, 1, 1, _ex(5.0333), IRF_CFG],
                ["ir_k5_sehsig_hs", 112, 1, 1, _ex(2.5070), IRF_CFG],
                ["ir_k5_sehsig_hs", 112, 1, 1, _ex(1.7712), IRF_CFG],
                ["ir_k5_sehsig_hs", 112, 1, 1, _ex(2.7712), IRF_CFG],
                ["ir_k5_sehsig_hs", 112, 1, 1, _ex(3.7712), IRF_CFG],
                ["ir_k5_sehsig_hs", 112, 1, 1, _ex(3.7712), IRF_CFG],
            ],
            [
                ["ir_k3_sehsig_hs", 184, 2, 1, _ex(5.5685), IRF_CFG],
                ["ir_k5_sehsig_hs", 184, 1, 1, _ex(2.8400), IRF_CFG],
                ["ir_k5_sehsig_hs", 184, 1, 1, _ex(2.8400), IRF_CFG],
                ["ir_k5_sehsig_hs", 184, 1, 1, _ex(4.8754), IRF_CFG],
                ["ir_k5_sehsig_hs", 184, 1, 1, _ex(4.8754), IRF_CFG],
                ["skip", 224, 1, 1, _ex(6.5245), IRF_CFG],
            ],
            [["ir_pool_hs", 1984, 1, 1, e6]],
        ],
    },
    "dmasking_l3": {
        # nparams: 9.402096, nflops 750.681952
        "input_size": 288,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 24, 2, 1]],
            # stage 1
            [["ir_k3", 24, 1, 1, e1, IRF_CFG]],
            # stage 2
            [
                ["ir_k5", 32, 2, 1, _ex(5.4566), IRF_CFG],
                ["ir_k5", 32, 1, 1, _ex(1.7912), IRF_CFG],
                ["ir_k3", 32, 1, 1, _ex(1.7912), IRF_CFG],
                ["ir_k5", 32, 1, 1, _ex(1.7912), IRF_CFG],
            ],
            # stage 3
            [
                ["ir_k5_sehsig", 48, 2, 1, _ex(5.3501), IRF_CFG],
                ["ir_k5_sehsig", 40, 1, 1, _ex(3.5379), IRF_CFG],
                ["ir_k5_sehsig", 40, 1, 1, _ex(4.5379), IRF_CFG],
                ["ir_k5_sehsig", 40, 1, 1, _ex(4.5379), IRF_CFG],
            ],
            # stage 4
            [
                ["ir_k5_hs", 72, 2, 1, _ex(5.7133), IRF_CFG],
                ["ir_k3_hs", 72, 1, 1, _ex(2.1212), IRF_CFG],
                ["skip", 72, 1, 1, _ex(3.1246), IRF_CFG],
                ["ir_k3_hs", 72, 1, 1, _ex(3.1246), IRF_CFG],
                ["ir_k3_hs", 120, 1, 1, _ex(5.0333), IRF_CFG],
                ["ir_k5_sehsig_hs", 120, 1, 1, _ex(2.5070), IRF_CFG],
                ["ir_k5_sehsig_hs", 120, 1, 1, _ex(1.7712), IRF_CFG],
                ["ir_k5_sehsig_hs", 120, 1, 1, _ex(2.7712), IRF_CFG],
                ["ir_k5_sehsig_hs", 120, 1, 1, _ex(3.7712), IRF_CFG],
                ["ir_k5_sehsig_hs", 120, 1, 1, _ex(3.7712), IRF_CFG],
                ["ir_k5_sehsig_hs", 120, 1, 1, _ex(3.7712), IRF_CFG],
            ],
            # stage 5
            [
                ["ir_k3_sehsig_hs", 192, 2, 1, _ex(5.5685), IRF_CFG],
                ["ir_k5_sehsig_hs", 192, 1, 1, _ex(2.8400), IRF_CFG],
                ["ir_k5_sehsig_hs", 192, 1, 1, _ex(2.8400), IRF_CFG],
                ["ir_k5_sehsig_hs", 192, 1, 1, _ex(4.8754), IRF_CFG],
                ["ir_k5_sehsig_hs", 192, 1, 1, _ex(4.8754), IRF_CFG],
                ["skip", 240, 1, 1, _ex(6.5245), IRF_CFG],
            ],
            # stage 6
            [["ir_pool_hs", 1984, 1, 1, e6]],
        ],
    },
}


def parse_block_cfg(block_op, out_channels, stride=1, repeat=1, *args):
    assert all(isinstance(x, dict) for x in args), f"{args}"
    cfg = {"out_channels": out_channels, "stride": stride}
    [update_dict(cfg, x) for x in args]

    ret = {"block_op": block_op, "block_cfg": cfg, "repeat": repeat}

    return ret


def parse_block_cfgs(block_cfgs):
    """ Parse block_cfgs like
            [
                [
                    ("ir_k3", 32, 2, 1)
                ],
                [
                    (
                        "ir_k3", 32, 2, 2,
                        {"expansion": 6, "dw_skip_bnrelu": True},
                        {"width_divisor": 8}
                    ),
                    ["conv_k1", 16, 1, 1]
                ],
            ]
        to:
            [
                [
                    {
                        "block_op": "ir_k3",
                        "block_cfg": {"out_channels": 32, "stride": 2}
                        "repeat: 1,
                    }
                ],
                [
                    {
                        "block_op": "ir_k3",
                        "block_cfg": {
                            "out_channels": 32, "stride": 2,
                            "expansion": 6, "dw_skip_bnrelu": True,
                            "width_divisor": 8
                        },
                        "repeat": 2,
                    },
                    {
                        "block_op": "conv_k1",
                        "block_cfg": {"out_channels": 16, "stride": 1},
                        "repeat": 1,
                    },
                ]
            ]
        The optional cfgs in each block (dicts) will be merged together in the
          order they appear in the dict.
    """
    assert isinstance(block_cfgs, list)
    ret = []
    for stage_cfg in block_cfgs:
        cur_stage = []
        for block_cfg in stage_cfg:
            assert isinstance(block_cfg, (list, tuple))
            cur_block = parse_block_cfg(*block_cfg)
            cur_stage.append(cur_block)
        ret.append(cur_stage)
    return ret


def _check_is_list(obj):
    assert isinstance(obj, (tuple, list)), f"{obj} is not a list"


def _check_lists_equal_size(*args):
    if len(args) == 0:
        return
    [_check_is_list(x) for x in args]
    size = len(args[0])
    assert all(len(x) == size for x in args), f"{args}"


def expand_repeats(blocks_info):
    """ Expand repeats in block cfg to multiple blocks and remove `_repeat_`
        Special handling for stride when repeat > 1 that the additionally expanded
            blocks will have stride 1
    """
    _check_is_list(blocks_info)
    ret = []
    for stage_cfgs in blocks_info:
        _check_is_list(stage_cfgs)
        cur_stage = []
        for block_cfg in stage_cfgs:
            assert isinstance(block_cfg, dict) and "block_cfg" in block_cfg
            cur_cfg = copy.deepcopy(block_cfg)
            repeat = cur_cfg.pop("repeat", 1)
            assert repeat >= 0
            # skip the block if repeat == 0
            if repeat == 0:
                continue
            expanded_cfgs = [copy.deepcopy(cur_cfg) for _ in range(repeat)]
            stride = cur_cfg["block_cfg"].get("stride", None)
            if repeat > 1 and stride is not None:
                # setup all strides to 1 except the first block
                for cur in expanded_cfgs[1:]:
                    cur["block_cfg"]["stride"] = 1
            cur_stage += expanded_cfgs
        ret.append(cur_stage)
    return ret


def flatten_stages(blocks_info):
    """ Flatten the blocks info from a list of list to a list
        Add 'stage_idx' and 'block_idx' to the blocks
    """
    _check_is_list(blocks_info)
    ret = []
    for stage_idx, stage_cfgs in enumerate(blocks_info):
        for block_idx, block_cfg in enumerate(stage_cfgs):
            cur = copy.deepcopy(block_cfg)
            cur["stage_idx"] = stage_idx
            cur["block_idx"] = block_idx
            ret.append(cur)
    return ret


def unify_arch_def_blocks(arch_def_blocks):
    """ unify an arch_def list
        [
            # [op, c, s, n, ...]
            # stage 0
            [("conv_k3", 32, 2, 1)],
            # stage 1
            [("ir_k3", 16, 1, 1, e1)],
        ]
        to
        [
            {
                "stage_idx": idx,
                "block_idx": idx,
                "block_cfg": {"out_channels": 32, "stride": 1, ...},
                "block_op": "conv_k3",
            },
            {}, ...
        ]
    """
    assert isinstance(arch_def_blocks, list)

    blocks_info = parse_block_cfgs(arch_def_blocks)
    blocks_info = expand_repeats(blocks_info)
    blocks_info = flatten_stages(blocks_info)

    return blocks_info


def unify_arch_def(arch_def, unify_names):
    """ unify an arch_def list
        {
            "blocks": [
                # [op, c, s, n, ...]
                # stage 0
                [("conv_k3", 32, 2, 1)],
                # stage 1
                [("ir_k3", 16, 1, 1, e1)],
            ]
        }
        to
        [
            "blocks": [
                {
                    "stage_idx": idx,
                    "block_idx": idx,
                    "block_cfg": {"out_channels": 32, "stride": 1, ...},
                    "block_op": "conv_k3",
                },
                {}, ...
            ],
        ]
    """
    assert isinstance(arch_def, dict)
    assert isinstance(unify_names, list)

    ret = copy.deepcopy(arch_def)
    for name in unify_names:
        if name not in ret:
            continue
        ret[name] = unify_arch_def_blocks(ret[name])

    return ret


def get_num_stages(arch_def_blocks):
    assert isinstance(arch_def_blocks, list)
    assert all("stage_idx" in x for x in arch_def_blocks)
    ret = 0
    for x in arch_def_blocks:
        ret = max(x["stage_idx"], ret)
    ret = ret + 1
    return ret


def get_stages_dim_out(arch_def_blocks):
    """ Calculates the output channels of stage_idx

    Assuming the blocks in a stage are ordered, returns the c of tcns in the
    last block of the stage by going through all blocks in arch def
    Inputs: (dict) architecutre definition
            (int) stage idx
    Return: (list of int) stage output channels
    """
    assert isinstance(arch_def_blocks, list)
    assert all("stage_idx" in x for x in arch_def_blocks)
    dim_out = [0] * get_num_stages(arch_def_blocks)
    for block in arch_def_blocks:
        stage_idx = block["stage_idx"]
        dim_out[stage_idx] = block["block_cfg"]["out_channels"]
    return dim_out


def get_num_blocks_in_stage(arch_def_blocks):
    """ Calculates the number of blocks in stage_idx

    Iterates over arch_def and counts the number of blocks
    Inputs: (dict) architecture definition
            (int) stage_idx
    Return: (list of int) number of blocks for each stage
    """
    assert isinstance(arch_def_blocks, list)
    assert all("stage_idx" in x for x in arch_def_blocks)
    nblocks = [0] * get_num_stages(arch_def_blocks)
    for block in arch_def_blocks:
        stage_idx = block["stage_idx"]
        nblocks[stage_idx] += 1
    return nblocks


def count_strides(arch_def_blocks):
    assert isinstance(arch_def_blocks, list)
    assert all("block_cfg" in x for x in arch_def_blocks)
    ret = 1
    for stride in count_stride_each_block(arch_def_blocks):
        ret *= stride
    return ret


def count_stride_each_block(arch_def_blocks):
    assert isinstance(arch_def_blocks, list)
    assert all("block_cfg" in x for x in arch_def_blocks)
    ret = []
    for block in arch_def_blocks:
        stride = block["block_cfg"]["stride"]
        assert stride != 0, stride
        if stride > 0:
            ret.append(stride)
        else:
            ret.append(1.0 / -stride)
    return ret


BLOCK_KWARGS_NAME = "block_kwargs"


def add_block_kwargs(block, kwargs):
    if kwargs is None:
        return
    if BLOCK_KWARGS_NAME not in block:
        block[BLOCK_KWARGS_NAME] = {}
    block[BLOCK_KWARGS_NAME].update(kwargs)


def get_block_kwargs(block):
    return block.get(BLOCK_KWARGS_NAME, None)


def update_with_block_kwargs(dest, block):
    block_kwargs = get_block_kwargs(block)
    if block_kwargs is not None:
        assert isinstance(block_kwargs, dict)
        dest.update(block_kwargs)
    return dest


class FBNetBuilder(object):
    def __init__(self, width_ratio=1.0, bn_args="bn", width_divisor=1):
        self.width_ratio = width_ratio
        self.last_depth = -1
        self.width_divisor = width_divisor
        # basic arguments that will be provided to all primitivies, they could be
        #   overrided by primitive parameters
        self.basic_args = {
            "bn_args": unify_args(bn_args),
            "width_divisor": width_divisor,
        }

    def add_basic_args(self, **kwargs):
        """ args that will be passed to all primitives, they could be
              overrided by primitive parameters
        """
        update_dict(self.basic_args, kwargs)

    def build_blocks(
            self,
            blocks,
            stage_indices=None,
            dim_in=None,
            prefix_name="xif",
            **kwargs,
    ):
        """ blocks: [{}, {}, ...]

        Inputs: (list(int)) stages to add
                (list(int)) if block[0] is not connected to the most
                            recently added block, list specifies the input
                            dimensions of the blocks (as self.last_depth
                            will be inaccurate)
        """
        assert isinstance(blocks, list) and all(
            isinstance(x, dict) for x in blocks
        ), blocks

        if stage_indices is not None:
            blocks = [x for x in blocks if x["stage_idx"] in stage_indices]
        if dim_in is not None:
            self.last_depth = dim_in
        assert (
                self.last_depth != -1
        ), "Invalid input dimension. Pass `dim_in` to `add_blocks`."

        modules = OrderedDict()
        for block in blocks:
            stage_idx = block["stage_idx"]
            block_idx = block["block_idx"]
            block_op = block["block_op"]
            block_cfg = block["block_cfg"]
            cur_kwargs = update_with_block_kwargs(copy.deepcopy(kwargs), block)
            nnblock = self.build_block(
                block_op, block_cfg, dim_in=None, **cur_kwargs
            )
            nn_name = f"{prefix_name}{stage_idx}_{block_idx}"
            assert nn_name not in modules
            modules[nn_name] = nnblock
        ret = nn.Sequential(modules)
        ret.out_channels = self.last_depth
        return ret

    def build_block(self, block_op, block_cfg, dim_in=None, **kwargs):
        if dim_in is None:
            dim_in = self.last_depth
        assert "out_channels" in block_cfg
        block_cfg = copy.deepcopy(block_cfg)
        out_channels = block_cfg.pop("out_channels")
        out_channels = self._get_divisible_width(
            out_channels * self.width_ratio
        )
        # dicts appear later will override the configs in the earlier ones
        new_kwargs = get_merged_dict(self.basic_args, block_cfg, kwargs)
        ret = PRIMITIVES.get(block_op)(dim_in, out_channels, **new_kwargs)
        self.last_depth = getattr(ret, "out_channels", out_channels)
        return ret

    def _get_divisible_width(self, width):
        ret = get_divisible_by(
            int(width), self.width_divisor, self.width_divisor
        )
        return ret


class ClsConvHead(nn.Module):
    """Global average pooling + conv head for classification
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        # global avg pool of arbitrary feature map size
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(input_dim, output_dim, 1)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return x


def _create_builder(arch_name_or_def: typing.Union[str, dict]):
    if isinstance(arch_name_or_def, str):
        assert arch_name_or_def in MODEL_ARCH_DMASKING_NET, (
            f"Invalid arch name {arch_name_or_def}, "
            f"available names: {MODEL_ARCH_DMASKING_NET.keys()}"
        )
        arch_def = MODEL_ARCH_DMASKING_NET[arch_name_or_def]
    else:
        assert isinstance(arch_name_or_def, dict)
        arch_def = arch_name_or_def

    arch_def = unify_arch_def(arch_def, ["blocks"])

    scale_factor = 1.0
    width_divisor = 1
    bn_info = {"name": "bn", "momentum": 0.003}
    drop_out = 0.2

    arch_def["dropout_ratio"] = drop_out

    builder = FBNetBuilder(
        width_ratio=scale_factor, bn_args=bn_info, width_divisor=width_divisor
    )
    builder.add_basic_args(**arch_def.get("basic_args", {}))

    return builder, arch_def


class FBNetBackbone(nn.Module):
    def __init__(self, arch_name, dim_in=3):
        super().__init__()

        builder, arch_def = _create_builder(arch_name)
        self.stages = builder.build_blocks(arch_def["blocks"], dim_in=dim_in)
        self.dropout = add_dropout(arch_def["dropout_ratio"])
        self.out_channels = builder.last_depth
        self.arch_def = arch_def

    def forward(self, x):
        y = self.stages(x)
        if self.dropout is not None:
            y = self.dropout(y)
        return y


class FBNet(nn.Module):
    def __init__(self, arch_name, dim_in=3, num_classes=1000):
        super().__init__()
        self.backbone = FBNetBackbone(arch_name, dim_in)
        self.head = ClsConvHead(self.backbone.out_channels, num_classes)

    def forward(self, x):
        y = self.backbone(x)
        y = self.head(y)
        return y

    @property
    def arch_def(self):
        return self.backbone.arch_def


def fbnetv2_f1(**kwargs):
    return FBNet("dmasking_f1")


def fbnetv2_f4(**kwargs):
    return FBNet("dmasking_f4")


def fbnetv2_l2_hs(**kwargs):
    return FBNet("dmasking_l2_hs")


def fbnetv2_l3(**kwargs):
    return FBNet("dmasking_l3")
