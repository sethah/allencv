import numpy as np
from typing import List, Tuple

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

    KEYPOINT_UNLABELED = 0
    KEYPOINT_LABELED_INVISIBLE = 1
    KEYPOINT_LABELED_VISIBLE = 2

    def __init__(self,
                 augmentation: List[ImageTransform] = None,
                 bbox_format: str = 'pascal_voc',
                 lazy: bool = False):
        super(ImageDatasetReader, self).__init__(lazy)
        if augmentation is None:
            _transforms = []
        else:
            _transforms = [a.transform for a in augmentation]
        self.augmentation = aug.Compose(_transforms,
                                        bbox_params={'format': bbox_format,
                                                     'label_fields': ['category_id']},
                                        keypoint_params={'format': 'xy', 'remove_invisible': False})

    def _unflatten_keypoints(self,
                             keypoints: List[Tuple[float, float]],
                             visibility: List[int],
                             lengths: List[int],
                             image_height: float,
                             image_width: float) -> List[List[Tuple[float, float, float]]]:
        """
        Keypoints need to be un-nested so they are per-object. And their visibilities need
        to be tacked back on, and potentially updated if augmentation moved them out
        of the image bounds.
        """
        _kp_visibility = []
        _kp_coords = []
        i = 0
        for l in lengths:
            _kp_coords.append(keypoints[i:i+l])
            _kp_visibility.append(visibility[i:i+l])
            i += l
        all_keypoints = []
        for object_kp_coords, object_kp_viz in zip(_kp_coords, _kp_visibility):
            ob_keypoints = []
            for (x, y), viz in zip(object_kp_coords, object_kp_viz):
                if viz == ImageDatasetReader.KEYPOINT_LABELED_VISIBLE:
                    # keypoints may now be out of the picture, which means invisible
                    if not self._keypoint_is_visible(image_width, image_height, x, y):
                        viz = ImageDatasetReader.KEYPOINT_UNLABELED
                ob_keypoints.append([x, y, viz])
            all_keypoints.append(ob_keypoints)
        return all_keypoints

    def _keypoint_is_visible(self, width: float, height: float, x: float, y: float) -> bool:
        return x >= 0 and x <= width and y >= 0 and y <= height

    def _flatten_keypoints(self,
                           nested_keypoints: List[List[Tuple[float, float, float]]]
                           ) -> Tuple[List[Tuple[float, float]], List[float], List[int]]:
        lengths = [len(kp_list) for kp_list in nested_keypoints]
        flat_keypoints = [item for sublist in nested_keypoints for item in sublist]
        keypoint_coords = [(kp[0], kp[1]) for kp in flat_keypoints]
        keypoint_visibility = [kp[2] for kp in flat_keypoints]
        return keypoint_coords, keypoint_visibility, lengths

    def augment(self,
                image: np.ndarray,
                masks: List[np.ndarray] = list(),
                boxes: List[np.ndarray] = list(),
                category_id: List[str] = list(),
                keypoints: List[List[Tuple[float, float, float]]] = list()):

        if keypoints:
            keypoints, keypoint_visibility, keypoint_lengths = \
                self._flatten_keypoints(keypoints)

        augmented = self.augmentation(image=image,
                                      masks=masks,
                                      bboxes=boxes,
                                      category_id=category_id,
                                      keypoints=keypoints)
        transformed_keypoints = augmented['keypoints']
        if keypoints:
            h, w = augmented['image'].shape[:2]
            keypoints = self._unflatten_keypoints(transformed_keypoints,
                                                          keypoint_visibility,
                                                          keypoint_lengths, w, h)
        return augmented['image'], augmented['masks'], augmented['bboxes'], \
               augmented['category_id'], keypoints
