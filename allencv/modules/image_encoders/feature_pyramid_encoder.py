from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from allencv.modules.image_encoders import ImageEncoder


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

    Parameters
    ----------
    backbone: ``ImageEncoder``
        Backbone that produces feature maps at various scales.
    output_channels: ``int``
        The number of output channels in each output feature map.
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
            self._convert_layers.append(nn.Conv2d(n_channels, self._output_channels,
                                                       kernel_size=1, stride=1))
            self._combine_layers.append(
                nn.Conv2d(self._output_channels, self._output_channels, kernel_size=3, stride=1,
                          padding=1))

    def forward(self, image: torch.Tensor) -> Sequence[torch.Tensor]:
        outputs = self._backbone.forward(image)

        n = len(outputs)
        converted_images = [self._convert_layers[-1].forward(outputs[-1])]
        out = converted_images[-1]
        for i in range(n - 2, -1, -1):
            converted = self._convert_layers[i].forward(outputs[i])
            upscaled = F.interpolate(out, size=(converted.shape[-2], converted.shape[-1]))
            out = converted + upscaled
            converted_images.append(out)
        combined_images = [layer.forward(im) for layer, im in
                           zip(self._combine_layers, converted_images)]
        combined_images.insert(0,
                               F.max_pool2d(combined_images[0], kernel_size=1, stride=2, padding=0))

        return list(reversed(combined_images))

    def get_input_channels(self) -> int:
        return self._backbone.get_input_channels()

    def get_output_channels(self) -> Sequence[int]:
        return [self._output_channels] * len(self._backbone.get_output_channels())

    def get_output_scales(self) -> Sequence[int]:
        return self._backbone.get_output_scales()

