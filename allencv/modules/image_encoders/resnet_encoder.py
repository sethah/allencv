from typing import Sequence, Union

import torch
from torchvision.models.resnet import resnet34, resnet50, resnet18, resnet101, resnet152, ResNet

from allencv.modules.image_encoders import ImageEncoder
import torch.nn as nn


@ImageEncoder.register("resnet_encoder")
class ResnetEncoder(ImageEncoder):
    """
        An ``ImageEncoder`` that passes an input image through a Resnet model structure.

        Parameters
        ----------
        resnet_model: ``ResNet``
            The base Resnet model.
        pretrained: ``bool``
            Use pretrained model from torchvision.
        requires_grad: ``bool``
            Shuts off backprop for this module.
        """

    def __init__(self,
                 resnet_model: Union[ResNet, str],
                 pretrained: bool = False,
                 requires_grad: bool = True,
                 norm_layer: nn.Module = None,
                 last_layer_maxpool: bool = False) -> None:
        super(ResnetEncoder, self).__init__()
        if isinstance(resnet_model, str):
            resnet_model = ResnetEncoder._pretrained_from_string(resnet_model,
                                                                 pretrained,
                                                                 requires_grad,
                                                                 norm_layer)
        self._input_channels = 3
        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4
        self._layers = [self.layer1, self.layer2, self.layer3, self.layer4]

    def forward(self, image: torch.Tensor) -> Sequence[torch.Tensor]:
        out = image
        out = self.maxpool(self.relu(self.bn1(self.conv1(out))))
        out_images = []
        for stage in self._layers:
            out = stage.forward(out)
            out_images.append(out)
        # TODO: not returning the first image here?
        return out_images

    def _get_test_outputs(self, height: int = 128, width: int = 128) -> Sequence[torch.Tensor]:
        params = list(self.parameters())
        im = torch.randn(1, self.get_input_channels(), height, width).to(params[0].device)
        return self.forward(im)

    def get_output_channels(self) -> Sequence[int]:
        return [x.shape[1] for x in self._get_test_outputs()]

    def get_input_channels(self) -> int:
        return 3

    def get_output_scales(self) -> Sequence[int]:
        test_height = 128
        test_width = 128
        outputs = self._get_test_outputs(test_height, test_width)
        return [test_height // out.shape[-2] for out in outputs]

    @staticmethod
    def _pretrained_from_string(resnet_string: str,
                                pretrained: bool = False,
                                requires_grad: bool = False,
                                norm_layer: nn.Module = None):
        if resnet_string == 'resnet34':
            model = resnet34(pretrained=pretrained, norm_layer=norm_layer)
        elif resnet_string == 'resnet18':
            model = resnet18(pretrained=pretrained, norm_layer=norm_layer)
        elif resnet_string == 'resnet152':
            model = resnet152(pretrained=pretrained, norm_layer=norm_layer)
        elif resnet_string == 'resnet50':
            model = resnet50(pretrained=pretrained, norm_layer=norm_layer)
        elif resnet_string == 'resnet101':
            model = resnet101(pretrained=pretrained, norm_layer=norm_layer)
        else:
            raise ValueError(f"Model {resnet_string} is not supported.")
        for param in model.parameters():
            param.requires_grad = requires_grad
        return model

