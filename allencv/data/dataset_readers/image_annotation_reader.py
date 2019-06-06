import json
import numpy as np
from overrides import overrides
from pathlib import Path
from PIL import Image

from typing import Dict, Iterable, List, Tuple
import logging

from allencv.data.dataset_readers.image_dataset_reader import ImageDatasetReader
from allencv.data.fields import ImageField, BoundingBoxField, KeypointField
from allencv.data.transforms.image_transform import ImageTransform

from allennlp.data.instance import Instance
from allennlp.data.fields import ArrayField, Field, LabelField, ListField
from allennlp.data.dataset_readers import DatasetReader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("image_annotation")
class ImageAnnotationReader(ImageDatasetReader):
    """
    Read pairs of images and annotations. Expects annotations in the Pascal VOC format.

    data_path
    |-- images
    |   |-- 00001.jpg
    |   |-- 00002.jpg
    |   |-- ...
    |-- annotations
    |   |-- 00001.json
    |   |-- 00002.json
    |   |-- ...

    Parameters
    ----------
    augmentation: ``list``, required
        List of image augmentations to apply to each pair.
    image_dir: ``str``, optional, (default='images')
        Name of the subdirectory which contains image inputs.
    annotation_dir: ``str``, optional, (default='images')
    annotation_ext: ``str``, optional
        File extension for the annotation images.
    """
    def __init__(self,
                 augmentation: List[ImageTransform] = None,
                 image_dir: str = "images",
                 annotation_dir: str = "annotations",
                 annotation_ext: str = ".json",
                 bbox_name: str = 'bbox',
                 class_name: str = 'class',
                 keypoint_name: str = None,
                 exclude_fields: List[str] = None,
                 num_keypoints: int = 1,
                 lazy: bool = False) -> None:
        super(ImageAnnotationReader, self).__init__(augmentation, lazy=lazy)
        self._image_dir = image_dir
        self._num_keypoints = num_keypoints
        self._annotation_dir = annotation_dir
        self._annotation_ext = annotation_ext
        self._bbox_name = bbox_name
        self._class_name = class_name
        self._keypoint_name = keypoint_name
        self.lazy = lazy
        if exclude_fields is None:
            self._exclude_fields = []
        else:
            self._exclude_fields = exclude_fields

    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = Path(file_path)

        for image_file in (file_path / self._image_dir).iterdir():
            img_name = image_file.stem
            img = Image.open(image_file)
            annotation_ext = self._annotation_ext
            annotation_file = file_path / self._annotation_dir / (img_name + annotation_ext)
            if not annotation_file.exists():
                annotations = {self._keypoint_name: None,
                               self._class_name: None,
                               self._bbox_name: None}
            else:
                with open(annotation_file, 'r') as f:
                    annotation = json.load(f)
                annotations = self._parse_annotation(annotation, self._num_keypoints)
            sample = img.convert('RGB')
            yield self.text_to_instance(np.array(sample),
                                        annotations.get(self._bbox_name, []),
                                        annotations.get(self._class_name, []),
                                        annotations.get(self._keypoint_name, []))

    def _parse_annotation(self, annotation, num_keypoints: int) -> Dict[str, List]:
        boxes = []
        classes = []
        keypoints = []
        for att in annotation['objects']:
            box = att['bbox']
            x, y, w, h = box
            if all([z == 0. for z in box]):
                continue
            if w <= 0 or h <= 0:
                continue
            x2, y2 = x + w, y + h
            if x > x2 or y > y2:
                continue
            boxes.append([x, y, x2, y2])
            classes.append(att['class'])
            kp = att.get("keypoints", [0, 0, 0] * num_keypoints)
            keypoints.append([kp[i:i+3] for i in range(0, len(kp), 3)])
        out = {}
        if self._class_name is not None:
            out[self._class_name] = classes
        if self._bbox_name is not None:
            out[self._bbox_name] = boxes
        if self._keypoint_name is not None:
            out[self._keypoint_name] = keypoints
        return out

    @overrides
    def text_to_instance(self,
                         image: np.ndarray,
                         label_box: List[List[float]] = list(),
                         label_class: List[str] = list(),
                         keypoints: List[List[Tuple[float, float, float]]] = list()) -> Instance:
        if label_box:
            img, _, label_box, label_class, keypoints = self.augment(image,
                                             boxes=[np.array(b) for b in label_box],
                                             category_id=label_class,
                                             keypoints=keypoints)
        else:
            img, _, _, _, _ = self.augment(image)
        h, w, c = img.shape
        fields: Dict[str, Field] = {}
        fields['image'] = ImageField(img.transpose(2, 0, 1), channels_first=False)
        fields['image_sizes'] = ArrayField(np.array([w, h]))
        if label_box:
            box_fields = [BoundingBoxField(x) for x in label_box]
            if 'boxes' not in self._exclude_fields:
                fields['boxes'] = ListField(box_fields)
            if 'box_classes' not in self._exclude_fields:
                fields['box_classes'] = ListField([LabelField(idx) for idx in label_class])
        if keypoints:
            if 'keypoint_positions' not in self._exclude_fields:
                fields['keypoint_positions'] = ListField([KeypointField(kp) for kp
                                                          in keypoints])
        return Instance(fields)

