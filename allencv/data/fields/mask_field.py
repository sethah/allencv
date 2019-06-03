from overrides import overrides
import numpy as np
from typing import Dict, Sequence, Tuple

import torch

from allencv.data.fields import ImageField


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
