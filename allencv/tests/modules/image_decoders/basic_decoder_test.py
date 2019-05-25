import torch

from allencv.common.testing import AllenCvTestCase

from allencv.modules.image_decoders import BasicDecoder


class TestBasicDecoder(AllenCvTestCase):

    def test_decoder_shapes(self):
        in_channels = 256
        out_channels = 128
        scales = [4, 2, 1]
        decoder = BasicDecoder(scales, in_channels, out_channels)
        input_sizes = [16, 32, 64]
        inputs = [torch.randn(1, in_channels, size, size) for size in input_sizes]
        out = decoder.forward(inputs)
        assert out.shape[-2:] == (64, 64)
        assert out.shape[1] == out_channels
        assert decoder.get_output_channels() == out_channels

