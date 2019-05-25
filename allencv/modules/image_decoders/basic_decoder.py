import numpy as np
from typing import List, Sequence

import torch
import torch.nn as nn

from allencv.nn import Upsample
from allencv.modules.image_decoders import ImageDecoder


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

