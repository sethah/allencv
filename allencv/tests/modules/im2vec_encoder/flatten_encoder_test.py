import torch
import torch.nn as nn

from allencv.common.testing import AllenCvTestCase

from allencv.modules.im2vec_encoders import FlattenEncoder

from allennlp.modules import FeedForward


class TestFlattenEncoder(AllenCvTestCase):

    def test_forward(self):
        in_channels = 256
        im_height = 28
        im_width = 28
        im = torch.randn(1, in_channels, im_height, im_width)
        feedforward = FeedForward(im_height * im_width * in_channels,
                                  num_layers=2, hidden_dims=64, activations=nn.ReLU())
        encoder = FlattenEncoder(in_channels, im_height, im_width)
        out = encoder.forward(im)
        assert out.shape == (1, 28 * 28 * 256)

        encoder_with_feedforward = FlattenEncoder(in_channels, im_height, im_width,
                                                  feedforward=feedforward)
        out_with_feedforward = encoder_with_feedforward.forward(im)
        assert out_with_feedforward.shape == (1, 64)


