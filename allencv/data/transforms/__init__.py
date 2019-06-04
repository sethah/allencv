from typing import Type
import numpy as np

from allencv.data.transforms.image_transform import ImageTransform

from allennlp.common import Params

import albumentations as aug
import albumentations.augmentations.functional as af
from albumentations.core.transforms_interface import BasicTransform, ImageOnlyTransform, DualTransform
import cv2


class _ImageTransformWrapper(object):

    def __init__(self, transform: Type[BasicTransform]):
        self._transform_class = transform

    def from_params(self, params: Params):
        transform = self._transform_class(**params.as_dict())
        return ImageTransform(transform)


class BGRNormalize(ImageOnlyTransform):

    def __init__(self,
                 mean=[102.9801, 115.9465, 122.7717],
                 std=[1., 1., 1.],
                 max_pixel_value=255.0,
                 always_apply=False, p=1.0):
        super(BGRNormalize, self).__init__(always_apply, p)
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value

    def apply(self, image, **params):
        image = image[:, :, [2, 1, 0]] * 255.
        return af.normalize(image, self.mean, self.std, self.max_pixel_value)


class KeypointResize(DualTransform):
    """Resize the input to the given height and width.

    Args:
        p (float): probability of applying the transform. Default: 1.
        height (int): desired height of the output.
        width (int): desired width of the output.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.

    Targets:
        image, mask, bboxes

    Image types:
        uint8, float32
    """

    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        super(KeypointResize, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        self.img_width = img.shape[1]
        self.img_height = img.shape[0]
        return af.resize(img, height=self.height, width=self.width, interpolation=interpolation)

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        x, y, a, s = keypoint
        return [x * self.width / self.img_width, y * self.height / self.img_height, a, s]



ImageTransform.register("resize")(_ImageTransformWrapper(aug.Resize))
ImageTransform.register("keypoint_resize")(_ImageTransformWrapper(KeypointResize))
ImageTransform.register("rotate")(_ImageTransformWrapper(aug.Rotate))
ImageTransform.register("gaussian_blur")(_ImageTransformWrapper(aug.GaussianBlur))
ImageTransform.register("elastic_transform")(_ImageTransformWrapper(aug.ElasticTransform))
ImageTransform.register("grid_distortion")(_ImageTransformWrapper(aug.GridDistortion))
ImageTransform.register("random_bbox_safe_crop")(_ImageTransformWrapper(aug.RandomSizedBBoxSafeCrop))
ImageTransform.register("random_brightness_contrast")(_ImageTransformWrapper(aug.RandomBrightnessContrast))
ImageTransform.register("bgr_normalize")(_ImageTransformWrapper(BGRNormalize))
ImageTransform.register("smallest_max_size")(_ImageTransformWrapper(aug.SmallestMaxSize))
ImageTransform.register("flip")(_ImageTransformWrapper(aug.Flip))
ImageTransform.register("horizontal_flip")(_ImageTransformWrapper(aug.HorizontalFlip))
ImageTransform.register("channel_shuffle")(_ImageTransformWrapper(aug.ChannelShuffle))
ImageTransform.register("normalize")(_ImageTransformWrapper(aug.Normalize))