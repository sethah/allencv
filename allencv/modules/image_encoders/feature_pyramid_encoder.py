import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from allencv.modules.image_encoders import ImageEncoder
from allencv.modules.im2im_encoders.feedforward_encoder import StdConv


class SamePad2d(nn.Module):
    """
    Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__

@ImageEncoder.register("feature_pyramid")
class FPN(ImageEncoder):
    """
    A feature pyramid network.

    Given a list of feature maps at multiple spatial resolutions encoded  by a
    particular backbone encoder, FPN upscales each and merges it with higher level features.
    This produces a set of feature maps that have both coarse and fine-grained information
    with a large receptive field.

    The backbone must provide relative scale factors for its feature maps.

    Reference: https://arxiv.org/abs/1612.03144
    """
    def __init__(self,
                 backbone: ImageEncoder,
                 output_channels: int):
        super(FPN, self).__init__()
        self._backbone = backbone
        self._upscale_factors = [a / b for a, b in zip(self._backbone.get_output_scales()[1:],
                                                       self._backbone.get_output_scales()[:-1])]
        self._output_channels = output_channels
        self.layers = nn.ModuleDict()
        self._convert_layers = nn.ModuleList()
        self._combine_layers = nn.ModuleList()

        for i, n_channels in enumerate(self._backbone.get_output_channels()):
            self._convert_layers.append(StdConv(n_channels, self._output_channels,
                                                       kernel_size=1, stride=1, dropout=0.4,
                                                   padding=0))
            self._combine_layers.append(nn.Sequential(
                SamePad2d(kernel_size=3, stride=1),
                nn.Conv2d(self._output_channels, self._output_channels, kernel_size=3, stride=1)))

    def forward(self, image: torch.Tensor) -> Sequence[torch.Tensor]:
        outputs = self._backbone.forward(image)

        n = len(outputs)
        converted_images = [self._convert_layers[-1].forward(outputs[-1])]
        out = converted_images[-1]
        for i in range(n - 2, -1, -1):
            out = self._convert_layers[i].forward(outputs[i]) + \
                  F.interpolate(out, scale_factor=self._upscale_factors[i])
            converted_images.append(out)
        combined_images = [layer.forward(im) for layer, im in
                           zip(self._combine_layers, converted_images)]

        return list(reversed(combined_images))

    def get_input_channels(self) -> int:
        return self._backbone.get_input_channels()

    def get_output_channels(self) -> Sequence[int]:
        return [self._output_channels] * len(self._backbone.get_output_channels())

    def get_output_scales(self) -> Sequence[int]:
        return self._backbone.get_output_scales()
