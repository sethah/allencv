from overrides import overrides

import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, resnet34, resnet50, resnet18, resnet101, resnet152

from allencv.modules.im2im_encoders import Im2ImEncoder
from allencv.modules.image_encoders import ImageEncoder, ResnetEncoder


class FromImageEncoder(Im2ImEncoder):

    def __init__(self,
                 image_encoder: ImageEncoder,
                 index: int = -1) -> None:
        super(FromImageEncoder, self).__init__()
        self._encoder = image_encoder
        self._index = index

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        output = self._encoder(inputs)[self._index]
        return output

    @overrides
    def get_input_channels(self) -> int:
        return self._encoder.get_input_channels()

    @overrides
    def get_output_channels(self) -> int:
        return self._encoder.get_output_channels()[self._index]


@Im2ImEncoder.register("resnet")
class ResnetIm2ImEncoder(FromImageEncoder):

    def __init__(self,
                 resnet_model: str,
                 pretrained: bool = True,
                 do_last_layer_pooling: bool = False,
                 requires_grad: bool = True):
        encoder = ResnetEncoder(resnet_model,
                                pretrained=pretrained,
                                requires_grad=requires_grad,
                                do_last_layer_pooling=do_last_layer_pooling)
        super(ResnetIm2ImEncoder, self).__init__(encoder, -1)

