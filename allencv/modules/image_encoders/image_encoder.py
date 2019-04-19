from overrides import overrides
from typing import Sequence

import torch

from allennlp.common import Registrable

from allencv.modules.encoder_base import _EncoderBase


class ImageEncoder(_EncoderBase, Registrable):

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                image: torch.Tensor) -> Sequence[torch.Tensor]:
        raise NotImplementedError

    def get_input_channels(self) -> int:
        raise NotImplementedError

    def get_output_channels(self) -> Sequence[int]:
        raise NotImplementedError

    def get_output_scales(self) -> Sequence[int]:
        raise NotImplementedError