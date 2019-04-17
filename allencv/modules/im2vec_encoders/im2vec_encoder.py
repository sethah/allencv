from allencv.modules.encoder_base import _EncoderBase

from allennlp.common import Registrable


class Im2VecEncoder(_EncoderBase, Registrable):

    def get_input_channels(self) -> int:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError
