import json
import numpy as np
from overrides import overrides
from pathlib import Path
from PIL import Image

from typing import Dict, Iterable, List
import logging

from allencv.data.dataset_readers.image_dataset_reader import ImageDatasetReader
from allencv.data.fields.image_field import ImageField, BoundingBoxField
from allencv.data.transforms.image_transform import ImageTransform

from allennlp.data.instance import Instance
from allennlp.data.fields import ArrayField, Field, LabelField, ListField
from allennlp.data.dataset_readers import DatasetReader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("image_annotation")
class ImageAnnotationReader(ImageDatasetReader):
    """
    """
    def __init__(self,
                 augmentation: List[ImageTransform] = list(),
                 image_dir: str = "images",
                 annotation_dir: str = "annotations",
                 annotation_ext: str = ".json",
                 lazy: bool = False) -> None:
        super(ImageAnnotationReader, self).__init__(augmentation, lazy=lazy)
        self._image_dir = image_dir
        self._annotation_dir = annotation_dir
        self._annotation_ext = annotation_ext
        self.lazy = lazy

    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = Path(file_path)

        for image_file in (file_path / self._image_dir).iterdir():
            img_name = image_file.stem
            img = Image.open(image_file)
            annotation_ext = self._annotation_ext
            annotation_file = file_path / self._annotation_dir / (img_name + annotation_ext)
            if not annotation_file.exists():
                label_boxes = None
            else:
                with open(annotation_file, 'r') as f:
                    annotation = json.load(f)
                label_boxes: List[List[float]] = self._parse_annotation(annotation)
            sample = img.convert('RGB')
            yield self.text_to_instance(np.array(sample), label_boxes, [1] * len(label_boxes))

    def _parse_annotation(self, annotation) -> List[List[float]]:
        boxes = []
        for att in annotation['regions']:
            box = att['shape_attributes']
            x1, y1 = box['x'], box['y']
            w, h = box['width'], box['height']
            x2, y2 = x1 + w, y1 + h
            boxes.append([x1, y1, x2, y2])
        return boxes

    @overrides
    def text_to_instance(self,
                         image: np.ndarray,
                         label_box: List[List[float]] = None,
                         label_class: List[str] = None) -> Instance:
        if label_box is not None:
            img, _, label_box = self.augment(image, boxes=[np.array(b) for b in label_box])
        else:
            img, _, _ = self.augment(image)
        h, w, c = img.shape
        fields: Dict[str, Field] = {}
        fields['image'] = ImageField(img.transpose(2, 0, 1), channels_first=False)
        fields['image_sizes'] = ArrayField(np.array([w, h]))
        if label_box is not None:
            box_fields = [BoundingBoxField(x) for x in label_box]
            fields['boxes'] = ListField(box_fields)
            fields['box_classes'] = ListField([LabelField(idx, skip_indexing=True)
                                               for idx in label_class])
        return Instance(fields)
