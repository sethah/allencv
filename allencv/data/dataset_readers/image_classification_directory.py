import numpy as np
from overrides import overrides
from pathlib import Path
from PIL import Image

from typing import Dict, Iterable, List
import logging

import torch

from allencv.data.transforms.image_transform import ImageTransform
from allencv.data.fields.image_field import ImageField
from allencv.data.dataset_readers.image_dataset_reader import ImageDatasetReader

from allennlp.data.instance import Instance
from allennlp.data.fields import LabelField, Field
from allennlp.data.dataset_readers import DatasetReader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("image_classification_directory")
class ImageClassificationDirectory(ImageDatasetReader):
    """
    Reads images from directories where the directory name is the label name for
    all images contained within. Only works one level deep currently.

    Parameters
    ----------
    skip_label_indexing: ``bool``, optional (default = ``False``)
        Whether or not to skip label indexing. You might want to skip label indexing if your
        labels are numbers, so the dataset reader doesn't re-number them starting from 0.
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be read lazily.
    """
    def __init__(self,
                 augmentation: List[ImageTransform],
                 skip_label_indexing: bool = False,
                 lazy: bool = False) -> None:
        super(ImageClassificationDirectory, self).__init__(augmentation)
        self.lazy = lazy
        self._skip_label_indexing = skip_label_indexing

    def _read(self, file_path: str) -> Iterable[Instance]:
        for label_dir in Path(file_path).iterdir():
            label = label_dir.name
            for img_file in label_dir.iterdir():
                if img_file.suffix not in ['.jpg', '.png']:
                    continue
                with open(img_file, 'rb') as f:
                    try:
                        img = Image.open(f)
                    except (UnboundLocalError, OSError) as e:
                        logger.error(f"Couldn't read {img_file}")
                        continue
                    sample = img.convert('RGB')
                    yield self.text_to_instance(np.array(sample), label)

    @overrides
    def text_to_instance(self, image: np.ndarray, label: str = None) -> Instance:
        image, _, _ = self.augment(image)
        fields: Dict[str, Field] = {}
        fields['image'] = ImageField(image.transpose(2, 0, 1), channels_first=False)
        if label is not None:
            fields['label'] = LabelField(label, skip_indexing=self._skip_label_indexing)
        return Instance(fields)
