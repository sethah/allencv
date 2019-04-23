# pylint: disable=no-self-use,invalid-name
import pytest
import torch

from allennlp.common.testing import AllenNlpTestCase

from allencv.modules.image_encoders import ResnetEncoder, FPN


class TestFeaturePyramid(AllenNlpTestCase):

    def test_decoder_shapes(self):
        image = torch.randn(1, 3, 224, 224)
        backbone = ResnetEncoder.from_string(resnet_model='resnet34')
        output_channels = 128
        fpn = FPN(backbone, output_channels)
        assert fpn.get_output_scales() == [4, 8, 16, 32]
        assert fpn.get_input_channels() == backbone.get_input_channels()
        assert fpn.get_output_channels() == [output_channels] * 4
        encoded = fpn.forward(image)
        assert len(encoded) == 4
        assert [im.shape[-2] for im in encoded] == [56, 28, 14, 7]
        assert [im.shape[-1] for im in encoded] == [56, 28, 14, 7]
        assert all([im.shape[1] == output_channels for im in encoded])


