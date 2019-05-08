from typing import Sequence

import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, resnet34, resnet50, resnet18, resnet101, resnet152

from allencv.modules.image_encoders import ImageEncoder


@ImageEncoder.register("resnet_encoder")
class ResnetEncoder(ImageEncoder):
    """
        An ``ImageEncoder`` that passes an input image through a Resnet model structure.

        Parameters
        ----------
        resnet_model: ``ResNet``
            The base Resnet model.
        """

    def __init__(self,
                 resnet_model: ResNet) -> None:
        super(ResnetEncoder, self).__init__()
        self._input_channels = 3
        res_layers = [resnet_model.layer1, resnet_model.layer2,
                      resnet_model.layer3, resnet_model.layer4]
        self._output_channels = res_layers[1][0].conv1.out_channels
        self.stages = nn.ModuleList()
        self.stages.append(nn.Sequential(*[resnet_model.conv1,
                                      resnet_model.bn1,
                                      resnet_model.relu,
                                      resnet_model.maxpool, resnet_model.layer1]))
        self.stages.extend(res_layers[1:])

    def forward(self, image: torch.Tensor) -> Sequence[torch.Tensor]:
        out = image
        out_images = []
        for stage in self.stages:
            out = stage.forward(out)
            out_images.append(out)
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

    @classmethod
    def from_string(cls, resnet_model: str, pretrained: bool = False, requires_grad: bool = False):
        if resnet_model == 'resnet34':
            model = resnet34(pretrained=pretrained)
        elif resnet_model == 'resnet18':
            model = resnet18(pretrained=pretrained)
        elif resnet_model == 'resnet152':
            model = resnet152(pretrained=pretrained)
        elif resnet_model == 'resnet50':
            model = resnet50(pretrained=pretrained)
        elif resnet_model == 'resnet101':
            model = resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Model {resnet_model} is not supported.")
        for param in model.parameters():
            param.requires_grad = requires_grad
        return cls(model)


@ImageEncoder.register("pretrained_resnet")
class PretrainedResnetEncoder(ResnetEncoder):
    """
    Parameters
    ----------
    resnet_model: ``str``
        Name of the pretrained Resnet variant.
    requires_grad: ``bool``, optional (default = ``False``)
        Whether to continue training the Resnet model.
    """
    def __init__(self, resnet_model: str, requires_grad: bool = False):
        if resnet_model == 'resnet34':
            model = resnet34(pretrained=True)
        elif resnet_model == 'resnet18':
            model = resnet18(pretrained=True)
        elif resnet_model == 'resnet152':
            model = resnet152(pretrained=True)
        elif resnet_model == 'resnet50':
            model = resnet50(pretrained=True)
        elif resnet_model == 'resnet101':
            model = resnet101(pretrained=True)
        else:
            raise ValueError(f"Model {resnet_model} is not supported.")

        for param in model.parameters():
            param.requires_grad = requires_grad

        super(PretrainedResnetEncoder, self).__init__(model)
