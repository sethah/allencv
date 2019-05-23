# pylint: disable=no-self-use,invalid-name
import torch

from allennlp.common.testing import AllenNlpTestCase

from allencv.modules.image_encoders import ResnetEncoder, FPN


class TestFeaturePyramid(AllenNlpTestCase):

    def test_encoder_shapes(self):
        image = torch.randn(1, 3, 224, 224)
        backbone = ResnetEncoder(model_str='resnet101')
        output_channels = 128
        fpn = FPN(backbone, output_channels)
        assert fpn.get_output_scales() == [4, 8, 16, 32]
        assert fpn.get_input_channels() == backbone.get_input_channels()
        assert fpn.get_output_channels() == [output_channels] * 4
        encoded = fpn.forward(image)
        assert len(encoded) == 5
        assert [im.shape[-2] for im in encoded] == [56, 28, 14, 7, 4]
        assert [im.shape[-1] for im in encoded] == [56, 28, 14, 7, 4]
        assert all([im.shape[1] == output_channels for im in encoded])
        image = torch.randn(1, 3, 500, 500)
        encoded = fpn.forward(image)
        assert [im.shape[-2] for im in encoded] == [125, 63, 32, 16, 8]
        assert [im.shape[-1] for im in encoded] == [125, 63, 32, 16, 8]


