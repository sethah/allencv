import torch

from allencv.common.testing import AllenCvTestCase

from allencv.modules.im2im_encoders import ResnetIm2ImEncoder


class TestResnetEncoder(AllenCvTestCase):

    def test_forward(self):
        batch_size = 1
        in_channels = 3
        im_height = 224
        im_width = 224
        im = torch.randn(batch_size, in_channels, im_height, im_width)
        encoder = ResnetIm2ImEncoder('resnet18')
        out = encoder.forward(im)
        assert out.shape == (batch_size, encoder.get_output_channels(), 7, 7)

        encoder_with_pooling = ResnetIm2ImEncoder('resnet18', do_last_layer_pooling=True)
        out_with_pooling = encoder_with_pooling.forward(im)
        assert out_with_pooling.shape == (batch_size, encoder.get_output_channels(), 1, 1)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            ResnetIm2ImEncoder('resnet72')


