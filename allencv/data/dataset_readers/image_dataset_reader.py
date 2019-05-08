import numpy as np
from typing import List

from allencv.data.transforms.image_transform import ImageTransform
from allennlp.data.dataset_readers import DatasetReader

import albumentations as aug


class ImageDatasetReader(DatasetReader):
    """
    A dataset reader that includes an augmentation method. This method will automatically
    identify all images and bounding boxes which may need to be transformed.

    Parameters
    ----------
    augmentation: ``ImageTransform``s to be applied in order to the data.
    bbox_format: ``str``, optional, (default='pascal_voc')
        How to interpret bounding box coordinates: either xyxy ('pascal_voc') or xywh ('coco')
    """

    def __init__(self,
                 augmentation: List[ImageTransform],
                 bbox_format: str = 'pascal_voc',
                 lazy: bool = False):
        super(ImageDatasetReader, self).__init__(lazy)
        self.augmentation = aug.Compose([a.transform for a in augmentation],
                                        bbox_params={'format': bbox_format,
                                                     'label_fields': ['category_id']})

    def augment(self,
                 image: np.ndarray,
                 masks: List[np.ndarray] = list(),
                 boxes: List[np.ndarray] = list()):
        augmented = self.augmentation(image=image, masks=masks,
                                      bboxes=boxes, category_id=[0]*len(boxes))
        return augmented['image'], augmented['masks'], augmented['bboxes']
