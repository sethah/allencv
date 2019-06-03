from overrides import overrides
import numpy as np
from typing import Dict, Sequence, Tuple

import torch

from allennlp.data.fields import Field


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
