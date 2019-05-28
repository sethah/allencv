from collections import OrderedDict
from typing import Sequence, Union

import torch
from torchvision.models.resnet import conv1x1, conv3x3, resnet34, resnet50, resnet18, resnet101, resnet152, ResNet

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
                 requires_grad: bool = True) -> None:
        super(ResnetEncoder, self).__init__()
        if isinstance(resnet_model, str):
            resnet_model = ResnetEncoder._pretrained_from_string(resnet_model, pretrained, requires_grad)
        self._input_channels = 3
        # TODO: frozen batchnorm?
        self.stem = nn.Sequential(OrderedDict(
                [
                    ('conv1', resnet_model.conv1),
                    ('bn1', resnet_model.bn1),
                    ('relu1', resnet_model.relu),
                    ('maxpool', resnet_model.maxpool)
                ]))
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4
        self._layers = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]

    def forward(self, image: torch.Tensor) -> Sequence[torch.Tensor]:
        out = image
        out_images = []
        for stage in self._layers:
            out = stage.forward(out)
            out_images.append(out)
        # TODO: not returning the first image here?
        return out_images[1:]

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
                                requires_grad: bool = False):
        if resnet_string == 'resnet34':
            model = resnet34(pretrained=pretrained)
        elif resnet_string == 'resnet18':
            model = resnet18(pretrained=pretrained)
        elif resnet_string == 'resnet152':
            model = resnet152(pretrained=pretrained)
        elif resnet_string == 'resnet50':
            model = resnet50(pretrained=pretrained)
        elif resnet_string == 'resnet101':
            model = resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Model {resnet_string} is not supported.")
        for param in model.parameters():
            param.requires_grad = requires_grad
        return model


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

