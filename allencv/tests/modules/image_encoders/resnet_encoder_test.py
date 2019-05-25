# pylint: disable=no-self-use,invalid-name
import torch

from allennlp.common.testing import AllenNlpTestCase

from allencv.modules.image_encoders import ResnetEncoder


class TestResnetEncoder(AllenNlpTestCase):

    def test_encoder_shapes(self):
        image = torch.randn(1, 3, 224, 224)
        encoder = ResnetEncoder(model_str='resnet101')
        assert encoder.get_output_scales() == [4, 8, 16, 32]
        assert encoder.get_input_channels() == 3
        assert encoder.get_output_channels() == [256, 512, 1024, 2048]
        encoded = encoder.forward(image)
        assert len(encoded) == 4
        assert [im.shape[-2] for im in encoded] == [56, 28, 14, 7]
        assert [im.shape[-1] for im in encoded] == [56, 28, 14, 7]
        assert [im.shape[1] for im in encoded] == encoder.get_output_channels()
        image = torch.randn(1, 3, 500, 500)
        encoded = encoder.forward(image)
        assert [im.shape[-2] for im in encoded] == [125, 63, 32, 16]
        assert [im.shape[-1] for im in encoded] == [125, 63, 32, 16]


