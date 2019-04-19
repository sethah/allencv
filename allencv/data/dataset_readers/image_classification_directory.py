import numpy as np
from overrides import overrides
from pathlib import Path
from PIL import Image

from typing import Dict, Iterable, List
import logging

import torch

from allencv.data.transforms.image_transform import ImageTransform
from allencv.data.fields.image_field import ImageField, MaskField, BoundingBoxField

from allennlp.data.instance import Instance
from allennlp.data.fields import LabelField, Field
from allennlp.data.dataset_readers import DatasetReader

import albumentations as aug

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ImageDatasetReader(DatasetReader):
    """
    A dataset reader that includes an augmentation method. This method will automatically
    identify all images and bounding boxes which may need to be transformed.

    Parameters
    ----------
    augmentation: ``ImageTransform``s to be applied in order to the data.
    """

    def __init__(self, augmentation: List[ImageTransform], lazy: bool = False):
        super(ImageDatasetReader, self).__init__(lazy)
        self.augmentation = aug.Compose([a.transform for a in augmentation])

    def augment(self, instance: Instance) -> Instance:
        image_field = instance.fields['image']
        image = image_field.as_tensor(image_field.get_padding_lengths()).numpy()
        if image_field.channels_first:
            image = image.transpose(1, 2, 0)
        masks = []
        boxes = []
        new_fields = {}
        for field_name, field in instance.fields.items():
            if isinstance(field, MaskField):
                masks.append((field_name, field.as_tensor(field.get_padding_lengths()).numpy()))
            elif isinstance(field, BoundingBoxField):
                boxes.append((field_name, field.as_tensor(field.get_padding_lengths()).numpy()))
            else:
                new_fields[field_name] = field
        augmented = self.augmentation(image=image, masks=[mask for _, mask in masks],
                                      bboxes=[box for _, box in boxes])
        new_fields['image'] = ImageField(augmented['image'].transpose(2, 0, 1))
        for i, mask in enumerate(augmented['masks']):
            new_fields[masks[i][0]] = MaskField(mask)
        for i, box in enumerate(augmented['bboxes']):
            new_fields[boxes[i][0]] = torch.from_numpy(box)
        return Instance(new_fields)


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
                    yield self.augment(self.text_to_instance(sample, label))

    @overrides
    def text_to_instance(self, image: Image.Image, label: str = None) -> Instance:
        fields: Dict[str, Field] = {}
        fields['image'] = ImageField(np.array(image), channels_first=False)
        if label is not None:
            fields['label'] = LabelField(label, skip_indexing=self._skip_label_indexing)
        return Instance(fields)
