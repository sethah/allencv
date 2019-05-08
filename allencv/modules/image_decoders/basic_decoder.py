import numpy as np
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from allencv.modules.im2im_encoders.feedforward_encoder import StdConv
from allencv.modules.image_decoders import ImageDecoder


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


@ImageDecoder.register("basic")
class BasicDecoder(ImageDecoder):
    """
    Simple decoder that takes several encoded images, each with the same number of channels,
    upscales them to a particular size, and sums them. Each image is upscaled by a
    factor of 2 at a time, with convolution stages in between, until it reaches the desired size.

    Parameters
    ----------
    input_scales: ``List[int]``
        What factor to scale each image by in the forward pass. Must be powers of 2.
    input_channels: ``int``, the number of channels in each of the encoded images.
    output_channels: ``int``, output channels in the decoded image.
    """

    def __init__(self,
                 input_scales: List[int],
                 input_channels: int,
                 output_channels: int) -> None:
        super(BasicDecoder, self).__init__()
        up_scales = [int(np.log2(x)) for x in input_scales]
        self._output_channels = output_channels
        self.upsampling_stages = nn.ModuleList()
        for n in up_scales:
            stage = Upsample(input_channels, self._output_channels, num_stages=n, scale_factor=2)
            self.upsampling_stages.append(stage)

    def forward(self,  # type: ignore
                images: Sequence[torch.Tensor]) -> torch.Tensor:
        # pylint: disable=arguments-differ
        upsampled_output = 0.
        for upstage, image in zip(self.upsampling_stages, images):
            upsampled_output += upstage.forward(image)
        return upsampled_output

    def get_output_channels(self):
        return self._output_channels

