from overrides import overrides
import numpy as np
from typing import Dict, List, Tuple

import torch

from allennlp.data.fields import Field


class KeypointField(Field[np.ndarray]):

    def __init__(self, keypoints: List[Tuple[float, float, float]], padding_value: int = 0) -> None:
        self.keypoints = keypoints
        self.num_keypoints = len(keypoints)
        self.padding_value = padding_value

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {
            'classes': self.num_keypoints,
            'locations': 3
        }

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        return torch.tensor(self.keypoints, dtype=torch.float32)

    @overrides
    def empty_field(self):  # pylint: disable=no-self-use
        return KeypointField([[0, 0, 0]] * self.num_keypoints, padding_value=self.padding_value)

    def __str__(self) -> str:
        return f"KeypointField with num_keypoints: {self.num_keypoints}"