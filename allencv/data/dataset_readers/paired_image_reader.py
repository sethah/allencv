import numpy as np
from overrides import overrides
from pathlib import Path
from PIL import Image

from typing import Dict, Iterable, List
import logging

from allencv.data.dataset_readers.image_dataset_reader import ImageDatasetReader
from allencv.data.fields.image_field import ImageField, MaskField
from allencv.data.transforms.image_transform import ImageTransform

from allennlp.data.instance import Instance
from allennlp.data.fields import Field
from allennlp.data.dataset_readers import DatasetReader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("paired_image")
class PairedImageReader(ImageDatasetReader):
    """
    Read pairs of images as input and label. TODO: add expected structure

    Parameters
    ----------
    augmentation: ``list``, required
        List of image augmentations to apply to each pair.
    image_dir: ``str``, optional, (default='images')
        Name of the subdirectory which contains image inputs.
    mask_dir: ``str``, optional, (default='images')
    mask_ext: ``str``, optional
        File extension for the mask images.
    """
    def __init__(self,
                 augmentation: List[ImageTransform] = list(),
                 image_dir: str = "images",
                 mask_dir: str = "masks",
                 mask_ext: str = None,
                 lazy: bool = False) -> None:
        super(PairedImageReader, self).__init__(augmentation, lazy)
        self._image_dir = image_dir
        self._mask_dir = mask_dir
        self._mask_ext = mask_ext
        self.lazy = lazy

    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = Path(file_path)
        for image_file in (file_path / self._image_dir).iterdir():
            img_name = image_file.stem
            img = Image.open(image_file)
            mask_ext = self._mask_ext if self._mask_ext is not None else image_file.suffix
            mask_file = file_path / self._mask_dir / (img_name + mask_ext)
            mask = Image.open(mask_file)
            sample = img.convert('RGB')
            yield self.text_to_instance(np.array(sample), mask)

    @overrides
    def text_to_instance(self, image: np.ndarray, mask: np.ndarray = None) -> Instance:
        if mask is not None:
            img, masks, _ = self.augment(image, masks=[np.array(mask)])
            mask = masks[0]
        else:
            img, _, _ = self.augment(image)
        fields: Dict[str, Field] = {}
        fields['image'] = ImageField(img.transpose(2, 0, 1), channels_first=False)
        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask[:, :, np.newaxis]
            fields['mask'] = MaskField(mask, channels_first=False)
        return Instance(fields)
