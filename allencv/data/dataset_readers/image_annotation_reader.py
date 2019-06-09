import json
import numpy as np
from overrides import overrides
from pathlib import Path
from PIL import Image
import shutil
import tarfile
import tempfile

from typing import Dict, Iterable, List, Tuple
import logging

from allencv.data.dataset_readers.image_dataset_reader import ImageDatasetReader
from allencv.data.fields import ImageField, BoundingBoxField, KeypointField
from allencv.data.transforms.image_transform import ImageTransform

from allennlp.common.file_utils import cached_path
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
    bbox_name: ``str``
        Name of bounding box field in the annotations.
    bbox_class_name: ``str``
        Name of bounding box label field in the annotations.
    keypoint_name: ``str``
        Name of keypoint field in the annotations.
    bbox: ``str``
        Include bounding boxes in the output instance.
    bbox_class: ``str``
        Include bounding box class labels in the output instance.
    keypoints: ``str``
        Include keypoints in the output instance.
    num_keypoints: ``int``
        Number of keypoints in each bounding box.
    """
    def __init__(self,
                 augmentation: List[ImageTransform] = None,
                 image_dir: str = "images",
                 annotation_dir: str = "annotations",
                 annotation_ext: str = ".json",
                 bbox_name: str = 'bbox',
                 bbox_class_name: str = 'class',
                 keypoint_name: str = 'keypoints',
                 bbox: bool = True,
                 bbox_class: bool = True,
                 keypoints: bool = False,
                 num_keypoints: int = 1,
                 lazy: bool = False) -> None:
        super(ImageAnnotationReader, self).__init__(augmentation, lazy=lazy)
        self._image_dir = image_dir
        self._num_keypoints = num_keypoints
        self._annotation_dir = annotation_dir
        self._annotation_ext = annotation_ext
        self._bbox_name = bbox_name
        self._bbox_class_name = bbox_class_name
        self._keypoint_name = keypoint_name
        self.lazy = lazy
        include_fields = set()
        if bbox:
            include_fields.add(self._bbox_name)
        if keypoints:
            include_fields.add(self._keypoint_name)
        if bbox_class:
            include_fields.add(self._bbox_class_name)
        self._include_fields = include_fields

    def _read(self, file_path: str) -> Iterable[Instance]:
        if not Path(file_path).exists():
            filename = cached_path(str(file_path))
            tar = tarfile.open(filename, "r:gz")
            tar.extractall(filename + ".dir")
            tar.close()
            file_path = filename + ".dir"
        file_path = Path(file_path)

        for image_file in (file_path / self._image_dir).iterdir():
            img_name = image_file.stem
            img = Image.open(image_file)
            annotation_ext = self._annotation_ext
            annotation_file = file_path / self._annotation_dir / (img_name + annotation_ext)
            if not annotation_file.exists():
                annotations = {}
            else:
                with open(annotation_file, 'r') as f:
                    annotation = json.load(f)
                annotations = self._parse_annotation(annotation, self._num_keypoints)
            sample = img.convert('RGB')
            yield self.text_to_instance(np.array(sample),
                                        annotations.get(self._bbox_name, []),
                                        annotations.get(self._bbox_class_name, []),
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
        out[self._bbox_class_name] = classes
        out[self._bbox_name] = boxes
        out[self._keypoint_name] = keypoints
        return out

    @overrides
    def text_to_instance(self,
                         image: np.ndarray,
                         label_box: List[List[float]] = list(),
                         label_class: List[str] = list(),
                         keypoints: List[List[Tuple[float, float, float]]] = list()) -> Instance:
        if self._keypoint_name in self._include_fields:
            # protect against some augmentations not supporting keypoints
            img, _, label_box, label_class, keypoints = self.augment(image,
                                                 boxes=[np.array(b) for b in label_box],
                                                 category_id=label_class,
                                                 keypoints=keypoints)
        else:
            img, _, label_box, label_class, _ = self.augment(image,
                                                                 boxes=[np.array(b) for b in label_box],
                                                                 category_id=label_class)
        h, w, c = img.shape
        fields: Dict[str, Field] = {}
        fields['image'] = ImageField(img.transpose(2, 0, 1), channels_first=False)
        fields['image_sizes'] = ArrayField(np.array([w, h]))
        if self._bbox_name in self._include_fields and len(label_box) > 0:
            box_fields = [BoundingBoxField(x) for x in label_box]
            fields['boxes'] = ListField(box_fields)
        if self._bbox_class_name in self._include_fields and len(label_class) > 0:
            fields['box_classes'] = ListField([LabelField(idx) for idx in label_class])
        if self._keypoint_name in self._include_fields and len(keypoints) > 0:
            assert all([len(kp) == len(keypoints[0]) for kp in keypoints])
            fields['keypoint_positions'] = ListField([KeypointField(kp) for kp
                                                      in keypoints])
        return Instance(fields)

