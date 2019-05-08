# pylint: disable=no-self-use,invalid-name
import numpy as np
import torch

from allencv.common.testing.test_case import AllenCvTestCase
from allencv.data.fields import ImageField, MaskField


class TestImageField(AllenCvTestCase):

    def test_image_field(self):
        im = np.random.randn(3, 8, 10)
        im_field = ImageField(im)
        assert list(im_field.as_tensor(im_field.get_padding_lengths()).shape) == [3, 8, 10]
        padding = im_field.get_padding_lengths()
        assert padding['channels'] == 3
        assert padding['height'] == 8
        assert padding['width'] == 10

        im = np.random.randn(8, 10, 3)
        im_field = ImageField(im)
        assert list(im_field.as_tensor(im_field.get_padding_lengths()).shape) == [8, 10, 3]

    def test_mask_field(self):
        im = np.random.randn(3, 8, 10)
        mask_field = MaskField(im)
        assert list(mask_field.as_tensor(mask_field.get_padding_lengths()).shape) == [3, 8, 10]

        im = np.random.randn(1, 8, 10)
        mask_field = MaskField(im)
        assert list(mask_field.as_tensor(mask_field.get_padding_lengths()).shape) == [8, 10]

        im = np.random.randn(8, 10)
        mask_field = MaskField(im)
        assert list(mask_field.as_tensor(mask_field.get_padding_lengths()).shape) == [8, 10]

    def test_padding(self):
        im = np.random.randn(3, 8, 10)
        im_field = ImageField(im)
        im_tensor = im_field.as_tensor({'channels': 4, 'width': 11, 'height': 37})
        assert tuple(im_tensor.shape) == (4, 37, 11)
        self.assertTrue(torch.all(im_tensor[-1] == 0.).item())
        self.assertTrue(torch.all(im_tensor[:, 8:, :] == 0.).item())
        self.assertTrue(torch.all(im_tensor[:, :, 10:] == 0.).item())
