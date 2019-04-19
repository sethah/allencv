from overrides import overrides

import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, resnet34, resnet50, resnet18, resnet101, resnet152

from allencv.modules.im2im_encoders import Im2ImEncoder


class ResnetEncoder(Im2ImEncoder):
    """
    An ``Im2ImEncoder`` that passes an input image through a Resnet model structure.

    Parameters
    ----------
    resnet_model: ``ResNet``
        The base Resnet model.
    layers: ``int``
        The input will be passed through only the first ``layers`` layers of the
        base Resnet model.
    """
    def __init__(self,
                 resnet_model: ResNet,
                 layers: int = 4) -> None:
        super(ResnetEncoder, self).__init__()
        self._input_channels = 3
        res_layers = [resnet_model.layer1, resnet_model.layer2,
                      resnet_model.layer3, resnet_model.layer4]
        self._output_channels = res_layers[layers - 1][0].conv1.out_channels
        self.model = nn.Sequential(*([resnet_model.conv1,
                                    resnet_model.bn1,
                                    resnet_model.relu,
                                    resnet_model.maxpool] + res_layers[:layers] +
                                     [resnet_model.avgpool]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        output = self.model(inputs)
        return output

    @overrides
    def get_input_channels(self) -> int:
        return self._input_channels

    @overrides
    def get_output_channels(self) -> int:
        return self._output_channels


@Im2ImEncoder.register("pretrained_resnet")
class PretrainedResnetEncoder(ResnetEncoder):
    """
    Parameters
    ----------
    resnet_model: ``str``
        Name of the pretrained Resnet variant.
    requires_grad: ``bool``, optional (default = ``False``)
        Whether to continue training the Resnet model.
    num_layers: ``int``, optional (default = ``4``)
        How many of the 4 Resnet layers to include in the encoder.
    """
    def __init__(self, resnet_model: str, requires_grad: bool = False, num_layers: int = 4):
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

        super(PretrainedResnetEncoder, self).__init__(model, num_layers)
