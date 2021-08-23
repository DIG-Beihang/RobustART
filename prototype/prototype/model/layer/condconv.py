import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math


class CondConv2d(nn.Module):
    r"""CondConv: Conditionally Parameterized Convolutions for Efficient Inference
    https://papers.nips.cc/paper/8412-condconv-conditionally-parameterized-convolutions-for-efficient-inference.pdf

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        padding_mode (string, optional). Accepted values `zeros` and `circular` Default: `zeros`
        num_experts (int): Number of experts for mixture. Default: 1
        combine_kernel (bool):
            If ``True``, first combine kernels, then use the combined kernel for the forward;
            If ``False``, first forward with different kernels, then combine the transformed features.
            Default: False

    Examples::

        >>> m = CondConv2d(16, 32, 3, stride=2, num_experts=4, combine_kernel=False)
        >>> input = torch.randn(64, 16, 32, 32)
        >>> output = m(input)

    Note:
        Re-implementation by yuankun
        If you have questions, welcome to discuss

    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', num_experts=1, combine_kernel=False):
        super(CondConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.num_experts = num_experts
        self.combine_kernel = combine_kernel

        self.weight = Parameter(
            torch.Tensor(num_experts, out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(num_experts, out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def _combine_kernel_forward(self, input, weight, routing_weight):
        r""" Details:
        input: bs*c*h*w
        weight: (bs*oc)*c*kh*kw
        groups: bs
        """
        bs, _, h, w = input.size()
        k, oc, c, kh, kw = weight.size()
        input = input.view(1, -1, h, w)
        weight = weight.view(k, -1)
        new_weight = torch.mm(routing_weight, weight).view(-1, c, kh, kw)  # (bs*oc)*c*kh*kw
        if self.bias is not None:
            new_bias = torch.mm(routing_weight, self.bias).view(-1)
            output = F.conv2d(
                input=input, weight=new_weight, bias=new_bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups*bs)
        else:
            output = F.conv2d(
                input=input, weight=new_weight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups*bs)

        output = output.view(bs, oc, output.size(-2), output.size(-1))
        return output

    def _combine_feature_forward(self, input, weight, routing_weight):
        r""" Details:
        input: bs*(c*k)*h*w
        weight: (k*oc)*c*kh*kw
        groups: k
        """
        k, oc, c, kh, kw = weight.size()
        input = input.repeat(1, self.num_experts, 1, 1)
        weight = weight.view(-1, c, kh, kw)
        if self.bias is not None:
            bias = self.bias.view(-1)
            output = F.conv2d(
                input=input, weight=weight, bias=bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups*self.num_experts)
        else:
            output = F.conv2d(input=input, weight=weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*self.num_experts)

        bs, _, oh, ow = output.size()
        output = output.view(bs, self.num_experts, oc, oh, ow)
        routing_weight = routing_weight.view(bs, self.num_experts, 1, 1, 1)
        output = (output * routing_weight).sum(dim=1)
        return output

    def forward(self, input, routing_weight):
        if self.combine_kernel:
            return self._combine_kernel_forward(input, self.weight, routing_weight)
        else:
            return self._combine_feature_forward(input, self.weight, routing_weight)
