from typing import Tuple

from allencv.modules.encoder_base import _EncoderBase
from allennlp.common import Registrable


class Im2ImEncoder(_EncoderBase, Registrable):
    """
    A ``Im2ImEncoder`` is a ``Module`` that takes as input an image and returns a
    modified image. The height and width of the output image are variable depending on the
    dimensions of the input image.

    Input shape: ``(batch_size, input_channels, input_height, input_width)``;
    output shape: ``(batch_size, output_channels, output_height, output_width)``.
    """
    def get_input_channels(self) -> int:
        """
        Returns the number of input channels to this encoder.
        """
        raise NotImplementedError

    def get_output_channels(self) -> int:
        """
        Returns the number of output channels to this encoder.
        """
        raise NotImplementedError
