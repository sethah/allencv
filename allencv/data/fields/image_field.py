from overrides import overrides
import numpy as np
from PIL import Image
from typing import Dict, Sequence

import torch

from allennlp.data.fields import Field


class ImageField(Field[np.array]):
    """
    An ``ImageField`` stores an image a ``np.ndarray`` which must have exactly three
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
        assert len(image.shape) == 3
        self.image = image
        if not channels_first:
            self.image = self.image.transpose(2, 0, 1)
        self.padding_value = padding_value
        self.channels_first = channels_first

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {
            'channels': self.image.shape[0],
            'height': self.image.shape[1],
            'width': self.image.shape[2],
        }

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        np_img = self.image
        if not self.channels_first:
            np_img = self.image.transpose(1, 2, 0)
        return torch.from_numpy(np_img).float()

    @overrides
    def empty_field(self):  # pylint: disable=no-self-use
        # Pass the padding_value, so that any outer field, e.g., `ListField[ArrayField]` uses the
        # same padding_value in the padded ArrayFields
        return ImageField(np.empty(self.image.shape), padding_value=self.padding_value)

    def __str__(self) -> str:
        return f"ImageField with shape: {self.image.shape}."


class MaskField(ImageField):

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
