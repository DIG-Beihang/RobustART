import torch.nn as nn


class BasicRouter(nn.Module):
    r"""CondConv: Conditionally Parameterized Convolutions for Efficient Inference
    https://papers.nips.cc/paper/8412-condconv-conditionally-parameterized-convolutions-for-efficient-inference.pdf

    Args:
        c_in (int): Number of channels in the input image
        num_experts (int): Number of experts for mixture. Default: 1

    Examples::

        >>> m = BasicRouter(16, num_experts=4)
        >>> input = torch.randn(64, 16, 32, 32)
        >>> output = m(input)
    """

    def __init__(self, c_in, num_experts):
        super(BasicRouter, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(c_in, num_experts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
