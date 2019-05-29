from overrides import overrides
import numpy as np
from typing import Dict, Sequence

import torch

from allennlp.data.fields import Field


class ImageField(Field[np.array]):
    """
    An ``ImageField`` stores an image as a ``np.ndarray`` which must have exactly three
    dimensions.

    Parameters
    ----------
    image: ``np.ndarray``
    channels_first: ``bool``, optional (default=True)
        Whether the first or last dimension is the channel dimension.
    """
    def __init__(self,
                 image: np.ndarray,
                 channels_first: bool = True,
                 padding_value: int = 0) -> None:
        if len(image.shape) == 2:
            image = image[np.newaxis, :, :]
            channels_first = True
        self.image = image
        if not channels_first:
            h, w, c = self.image.shape
            self._layout = 'hwc'
        else:
            c, h, w = self.image.shape
            self._layout = 'chw'
        self._channels = c
        self._height = h
        self._width = w
        assert len(image.shape) == 3
        self.padding_value = padding_value
        self.channels_first = channels_first

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {
            'channels': self._channels,
            'height': self._height,
            'width': self._width
        }

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        np_img = self.image
        img = torch.from_numpy(np_img).float()
        if self._layout == 'hwc':
            pad_img = torch.zeros(padding_lengths['height'],
                                  padding_lengths['width'], padding_lengths['channels'])
        else:
            pad_img = torch.zeros(padding_lengths['channels'],
                              padding_lengths['height'], padding_lengths['width'])
        pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
        return pad_img

    @overrides
    def empty_field(self):  # pylint: disable=no-self-use
        return ImageField(np.empty(self.image.shape), padding_value=self.padding_value)

    def __str__(self) -> str:
        return f"ImageField with shape: {self.image.shape}."


class MaskField(ImageField):
    """
    You may want to use a ``MaskField`` to store an image that should be treated as a label
    instead of an input.
    """

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        np_img = self.image
        return torch.from_numpy(np_img).squeeze().float()

    def __str__(self) -> str:
        return f"MaskField with shape: {self.image.shape}."


class BoundingBoxField(Field[np.ndarray]):

    def __init__(self, coords: Sequence[float], padding_value: int = 0) -> None:
        self.coords = coords
        self.padding_value = padding_value

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {
            'coords': 4
        }

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        return torch.tensor(self.coords, dtype=torch.float32)

    @overrides
    def empty_field(self):  # pylint: disable=no-self-use
        return BoundingBoxField([0, 0, 0, 0], padding_value=self.padding_value)

    def __str__(self) -> str:
        return f"BoundingBoxField with coords: {self.coords}"
