import torch

from allennlp.common.checks import ConfigurationError

from allencv.common.testing import AllenCvTestCase
from allencv.modules.im2im_encoders import FeedforwardEncoder


class TestResnetEncoder(AllenCvTestCase):

    def test_forward(self):
        batch_size = 1
        in_channels = 3
        im_height = 224
        im_width = 224
        hidden_channels = [64, 128, 256]
        im = torch.randn(batch_size, in_channels, im_height, im_width)
        encoder = FeedforwardEncoder(input_channels=in_channels,
                                     num_layers=3,
                                     hidden_channels=hidden_channels,
                                     activations='relu')
        out = encoder.forward(im)
        assert out.shape == (batch_size, hidden_channels[-1], 224, 224)

        encoder_with_downsample = FeedforwardEncoder(input_channels=in_channels,
                                                     num_layers=3,
                                                     hidden_channels=hidden_channels,
                                                     activations='relu',
                                                     downsample=True)
        out_with_downsample = encoder_with_downsample.forward(im)
        assert out_with_downsample.shape == (batch_size, encoder.get_output_channels(), 28, 28)

    def test_invalid(self):
        with self.assertRaises(ConfigurationError):
            batch_size = 1
            in_channels = 3
            im_height = 224
            im_width = 224
            hidden_channels = [64, 128]
            im = torch.randn(batch_size, in_channels, im_height, im_width)
            encoder = FeedforwardEncoder(input_channels=in_channels,
                                         num_layers=3,
                                         hidden_channels=hidden_channels,
                                         activations='relu')
            encoder.forward(im)

