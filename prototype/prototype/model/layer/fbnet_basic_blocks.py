"""
FBNet model basic building blocks
"""

import logging
import numbers
import math

import torch
from torch.nn.modules.utils import _ntuple
import torch.nn as nn
from torch.nn.quantized.modules import FloatFunctional

from prototype.prototype.utils.fbnet_helper import _get_conv_2d_output_shape, merge, filter_kwargs, \
    merge_unify_args, unify_args, get_divisible_by, drop_connect_batch

logger = logging.getLogger(__name__)


# layers compatible with empty inputs

class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class GroupNorm(torch.nn.GroupNorm):

    def forward(self, x):
        if x.numel() > 0:
            return super(GroupNorm, self).forward(x)

        # get output shape
        output_shape = x.shape
        return _NewEmptyTensorOp.apply(x, output_shape)


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    if input.numel() > 0:
        return torch.nn.functional.interpolate(
            input, size, scale_factor, mode, align_corners
        )

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError("either size or scale_factor should be defined")
        if size is not None and scale_factor is not None:
            raise ValueError(
                "only one of size or scale_factor should be defined"
            )
        if (
            scale_factor is not None
            and isinstance(scale_factor, tuple)
            and len(scale_factor) != dim
        ):
            raise ValueError(
                "scale_factor shape must match input shape. "
                "Input is {}D, scale_factor size is {}".format(
                    dim, len(scale_factor)
                )
            )

    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            return size
        scale_factors = _ntuple(dim)(scale_factor)
        # math.floor might return float in py2.7
        return [
            int(math.floor(input.size(i + 2) * scale_factors[i]))
            for i in range(dim)
        ]

    output_shape = tuple(_output_size(2))
    output_shape = input.shape[:-2] + output_shape
    return _NewEmptyTensorOp.apply(input, output_shape)


class Conv2dEmptyOutput(torch.nn.Module):
    def __init__(self, conv_op):
        super().__init__()
        assert isinstance(conv_op, torch.nn.Conv2d)
        self.padding = conv_op.padding
        self.dilation = conv_op.dilation
        self.kernel_size = conv_op.kernel_size
        self.stride = conv_op.stride
        self.out_channels = conv_op.out_channels

    def forward(self, x):
        assert x.numel() == 0, "Only handle empty batch"
        output_shape = _get_conv_2d_output_shape(self, x)
        return _NewEmptyTensorOp.apply(x, output_shape)


class Identity(nn.Module):
    def __init__(self, in_channels, out_channels, stride, **kwargs):
        super().__init__()
        self.conv = None
        if in_channels != out_channels or stride != 1:
            self.conv = ConvBNRelu(
                in_channels,
                out_channels,
                **merge(
                    conv_args={
                        "kernel_size": 1,
                        "stride": stride,
                        "bias": False,
                    },
                    kwargs=kwargs,
                ),
            )
        self.out_channels = out_channels

    def forward(self, x):
        out = x
        if self.conv:
            out = self.conv(x)
        return out


class TorchAdd(nn.Module):
    """Wrapper around torch.add so that all ops can be found at build"""

    def __init__(self):
        super().__init__()
        self.add_func = FloatFunctional()

    def forward(self, x, y):
        return self.add_func.add(x, y)


class TorchAddScalar(nn.Module):
    """ Wrapper around torch.add so that all ops can be found at build
        y must be a scalar, needed for quantization
    """

    def __init__(self):
        super().__init__()
        self.add_func = FloatFunctional()

    def forward(self, x, y):
        return self.add_func.add_scalar(x, y)


class TorchMultiply(nn.Module):
    """Wrapper around torch.mul so that all ops can be found at build"""

    def __init__(self):
        super().__init__()
        self.mul_func = FloatFunctional()

    def forward(self, x, y):
        return self.mul_func.mul(x, y)


class TorchMulScalar(nn.Module):
    """Wrapper around torch.mul so that all ops can be found at build
        y must be a scalar, needed for quantization
    """

    def __init__(self):
        super().__init__()
        self.mul_func = FloatFunctional()

    def forward(self, x, y):
        return self.mul_func.mul_scalar(x, y)


class TorchCat(nn.Module):
    """Wrapper around torch.cat so that all ops can be found at build"""

    def __init__(self):
        super().__init__()
        self.cat_func = FloatFunctional()

    def forward(self, tensors, dim):
        return self.cat_func.cat(tensors, dim)


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert (
            C % g == 0
        ), "Incompatible group size {} for input channel {}".format(g, C)
        return (
            x.view(N, g, int(C / g), H, W)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(N, C, H, W)
        )


class HSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.clamp
        self.add_scalar = TorchAddScalar()
        self.mul_scalar = TorchMulScalar()

    def forward(self, x):
        return self.relu(self.add_scalar(x, 3.0), 0.0, 6.0) / 6.0


class HSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.hsig = HSigmoid()
        self.mul = TorchMultiply()

    def forward(self, x):
        return self.mul(x, self.hsig(x))


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sig = nn.Sigmoid()
        self.mul = TorchMultiply()

    def forward(self, x):
        return self.mul(x, self.sig(x))


def _init_conv_weight(op, weight_init="kaiming_normal"):
    assert weight_init in [None, "kaiming_normal"]
    if weight_init is None:
        return
    if weight_init == "kaiming_normal":
        nn.init.kaiming_normal_(op.weight, mode="fan_out", nonlinearity="relu")
        if hasattr(op, "bias") and op.bias is not None:
            nn.init.constant_(op.bias, 0.0)


def build_empty_input_op(op):
    """ Op to handle empty tensor input
        Return proper output tensor if input is an empty tensor
    """
    if op is None:
        return None
    if isinstance(op, nn.Conv2d):
        return Conv2dEmptyOutput(op)
    return None


def build_conv(
    name="conv",
    in_channels=None,
    out_channels=None,
    weight_init="kaiming_normal",
    **conv_args,
):
    if name is None:
        return None
    if name == "conv":
        conv_args = filter_kwargs(nn.Conv2d, conv_args)
        if "kernel_size" not in conv_args:
            conv_args["kernel_size"] = 1
        ret = nn.Conv2d(in_channels, out_channels, **conv_args)
        _init_conv_weight(ret, weight_init)
        return ret
    if name == "linear":
        ret = nn.Linear(in_channels, out_channels, bias=True)
        return ret

    return None


def build_bn(name, num_channels, zero_gamma=None, **bn_args):
    if name is None:
        bn_op = None
    elif name == "bn":
        bn_op = nn.BatchNorm2d(num_channels, **bn_args)
        if zero_gamma is True:
            nn.init.constant_(bn_op.weight, 0.0)
    elif name == "gn":
        bn_op = nn.GroupNorm(num_channels=num_channels, **bn_args)
    else:
        bn_op = None

    return bn_op


def build_relu(name=None, num_channels=None, **kwargs):
    if name is None:
        return None
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "relu6":
        return nn.ReLU6(inplace=True)
    if name == "leakyrelu":
        return nn.LeakyReLU(inplace=True)
    if name == "prelu":
        return nn.PReLU(num_parameters=num_channels, **kwargs)
    if name == "hswish":
        return HSwish()
    if name == "swish":
        return Swish()
    if name == "sig":
        return nn.Sigmoid()
    if name == "hsig":
        return HSigmoid()

    return None


class ConvBNRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_args="conv",
        bn_args="bn",
        relu_args="relu",
        # additional arguments for conv
        **kwargs,
    ):
        super().__init__()
        conv_op = build_conv(
            in_channels=in_channels,
            out_channels=out_channels,
            **merge_unify_args(conv_args, kwargs),
        )

        # register in order
        self.empty_input = build_empty_input_op(conv_op)
        self.conv = conv_op

        self.bn = (
            build_bn(num_channels=out_channels, **unify_args(bn_args))
            if bn_args is not None
            else None
        )
        self.relu = (
            build_relu(num_channels=out_channels, **unify_args(relu_args))
            if relu_args is not None
            else None
        )

        self.out_channels = out_channels

    def forward(self, x):
        if x.numel() > 0 or self.empty_input is None:
            if self.conv:
                x = self.conv(x)
            if self.bn:
                x = self.bn(x)
            if self.relu:
                x = self.relu(x)
        else:
            x = self.empty_input(x)
        return x


class SEModule(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        fc=False,
        sigmoid_type="sigmoid",
        relu_args="relu",
    ):
        super(SEModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if not fc:
            conv1_relu = ConvBNRelu(
                in_channels,
                mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bn_args=None,
                relu_args=relu_args,
            )
            conv2 = nn.Conv2d(mid_channels, in_channels, 1, 1, 0)
        else:
            conv1_relu = ConvBNRelu(
                in_channels,
                mid_channels,
                conv_args="linear",
                bn_args=None,
                relu_args=relu_args,
            )
            conv2 = nn.Linear(mid_channels, in_channels, bias=True)

        if sigmoid_type == "sigmoid":
            sig = nn.Sigmoid()
        elif sigmoid_type == "hsigmoid":
            sig = HSigmoid()
        else:
            raise Exception(f"Incorrect sigmoid_type {sigmoid_type}")

        self.se = nn.Sequential(conv1_relu, conv2, sig)
        self.use_fc = fc
        self.mul = TorchMultiply()

    def forward(self, x):
        n, c, _, _ = x.size()
        y = self.avg_pool(x)
        if self.use_fc:
            y = y.view(n, c)
        y = self.se(y)
        if self.use_fc:
            y = y.view(n, c, 1, 1).expand_as(x)
        return self.mul(x, y)


def build_se(
    name=None, in_channels=None, mid_channels=None, width_divisor=None, **kwargs
):
    if name is None:
        return None
    mid_channels = get_divisible_by(mid_channels, width_divisor)
    if name == "se":
        return SEModule(in_channels, mid_channels, **kwargs)
    if name == "se_fc":
        return SEModule(in_channels, mid_channels, fc=True, **kwargs)
    elif name == "se_hsig":
        return SEModule(
            in_channels, mid_channels, sigmoid_type="hsigmoid", **kwargs
        )
    raise Exception(f"Invalid SEModule arugments {name}")


class Upsample(nn.Module):
    def __init__(
        self, size=None, scale_factor=None, mode="nearest", align_corners=None
    ):
        super(Upsample, self).__init__()
        self.size = size
        self.scale = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return interpolate(
            x,
            size=self.size,
            scale_factor=self.scale,
            mode=self.mode,
            align_corners=self.align_corners,
        )

    def __repr__(self):
        ret = []
        attr_list = ["size", "scale", "mode", "align_corners"]
        for x in attr_list:
            val = getattr(self, x, None)
            if val is not None:
                ret.append(f"{x}={val}")
        return f"Upsample({', '.join(ret)})"


def build_upsample_neg_stride(name=None, stride=None, **kwargs):
    """ Use negative stride to represent scales, i.e., stride=-2 means scale=2
        Return upsample op if the stride is negative, return None otherwise
        Reset and return the stride to 1 if it is negative
    """
    if name is None:
        return None, stride

    if isinstance(stride, numbers.Number):
        stride = (stride, stride)
    assert isinstance(stride, (tuple, list))

    neg_strides = all(x < 0 for x in stride)
    if not neg_strides:
        return None, stride

    scales = [-x for x in stride]
    if name == "default":
        ret = Upsample(scale_factor=scales, **kwargs)
    else:
        ret = None

    return ret, 1


class AddWithDropConnect(nn.Module):
    """ Apply drop connect on x before adding with y """

    def __init__(self, drop_connect_rate):
        super().__init__()
        self.drop_connect_rate = drop_connect_rate
        self.add = TorchAdd()

    def forward(self, x, y):
        xx = drop_connect_batch(
            x, self.drop_connect_rate, self.training
        )
        return self.add(xx, y)

    def extra_repr(self):
        return f"drop_connect_rate={self.drop_connect_rate}"
