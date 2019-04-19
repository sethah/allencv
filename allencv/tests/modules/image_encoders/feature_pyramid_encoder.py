# pylint: disable=no-self-use,invalid-name
from numpy.testing import assert_almost_equal
import pytest
import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.modules import FeedForward
from allennlp.nn import InitializerApplicator, Initializer, Activation
from allennlp.common.testing import AllenNlpTestCase

from allencv.modules.image_encoders import BackboneEncoder, FPN


class TestEncoderDecoder(AllenNlpTestCase):

    def test_decoder_shapes(self):
        image = torch.randn(1, 3, 224, 224)
        backbone = BackboneEncoder()
        output_channels = 128
        fpn = FPN(backbone, output_channels)
        assert fpn.get_output_scales() == [4, 8, 16, 32]


