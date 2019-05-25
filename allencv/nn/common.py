import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.nn import Activation


class StdConv(nn.Module):

    def __init__(self,
                 nin: int,
                 nout: int,
                 kernel_size: int = 3,
                 activation: Activation = nn.ReLU(),
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 dropout: float = 0.1):
        super(StdConv, self).__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(nout)
        self.drop = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        return self.drop(self.bn(self.activation(self.conv(x))))


class Upsample(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_stages: int, scale_factor: int = 2):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.convs = [StdConv(in_channels, out_channels, padding=1)] + \
                [StdConv(out_channels, out_channels, padding=1) for _ in range(num_stages - 1)]
        self.convs = nn.ModuleList(self.convs)
        self.num_stages = num_stages

    def forward(self, inputs: torch.Tensor):
        out = inputs
        for conv in self.convs:
            out = conv.forward(out)
            if self.num_stages != 0:
                out = F.interpolate(out, scale_factor=self.scale_factor)
        return out
