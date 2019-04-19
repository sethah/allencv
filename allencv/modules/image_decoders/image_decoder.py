from overrides import overrides
from typing import Sequence

import torch

from allennlp.common import Registrable

from allencv.modules.decoder_base import _DecoderBase


class ImageDecoder(_DecoderBase, Registrable):

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                images: Sequence[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def get_output_channels(self) -> int:
        raise NotImplementedError