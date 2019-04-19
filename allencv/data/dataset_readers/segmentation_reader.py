import numpy as np
from overrides import overrides
from pathlib import Path
from PIL import Image

from typing import Dict, Iterable, List
import logging

import torch

from allencv.data.dataset_readers.image_classification_directory import ImageDatasetReader
from allencv.data.fields.image_field import ImageField, MaskField, BoundingBoxField
from allencv.data.transforms.image_transform import ImageTransform

from allennlp.data.instance import Instance
from allennlp.data.fields import LabelField, Field
from allennlp.data.dataset_readers import DatasetReader

import albumentations as aug

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("segmentation_reader")
class SegmentationReader(ImageDatasetReader):
    """
    """
    def __init__(self,
                 augmentation: List[ImageTransform],
                 skip_label_indexing: bool = False,
                 lazy: bool = False) -> None:
        super(SegmentationReader, self).__init__(augmentation, lazy)
        self.lazy = lazy
        self._skip_label_indexing = skip_label_indexing

    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = Path(file_path)
        for image_file in (file_path / "images").iterdir():
            img_name = image_file.stem
            img = Image.open(image_file)
            mask = Image.open(file_path / "masks" / (img_name + "_mask.gif"))
            sample = img.convert('RGB')
            yield self.augment(self.text_to_instance(sample, mask))

    @overrides
    def text_to_instance(self, image: Image.Image, label: Image.Image = None) -> Instance:
        fields: Dict[str, Field] = {}
        fields['image'] = ImageField(np.array(image), channels_first=False)
        if label is not None:
            mask = np.array(label)
            if len(mask.shape) == 2:
                mask = mask[:, :, np.newaxis]
            fields['label'] = MaskField(mask, channels_first=False)
        return Instance(fields)
