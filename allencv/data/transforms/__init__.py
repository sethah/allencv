from typing import Dict, Iterable, List, Type

from allencv.data.transforms.image_transform import ImageTransform

from allennlp.common import Params, Registrable

import albumentations as aug
from albumentations.core.transforms_interface import BasicTransform


class _ImageTransformWrapper(object):

    def __init__(self, transform: Type[BasicTransform]):
        self._transform_class = transform

    def from_params(self, params: Params):
        transform = self._transform_class(**params.as_dict())
        return ImageTransform(transform)

ImageTransform.register("resize")(_ImageTransformWrapper(aug.Resize))
ImageTransform.register("flip")(_ImageTransformWrapper(aug.Flip))
ImageTransform.register("channel_shuffle")(_ImageTransformWrapper(aug.ChannelShuffle))
ImageTransform.register("normalize")(_ImageTransformWrapper(aug.Normalize))