import math
from overrides import overrides
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torch_models

from allennlp.models import Model
from allencv.modules.encoder_base import _EncoderBase
from allencv.modules.decoder_base import _DecoderBase

from allennlp.common import Registrable

from allencv.modules.im2im_encoders import Im2ImEncoder
from allencv.modules.im2im_encoders.feedforward_encoder import StdConv

class SamePad2d(nn.Module):
    """
    Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__

class ImageEncoder(_EncoderBase, Registrable):

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                image: torch.Tensor) -> Sequence[torch.Tensor]:
        raise NotImplementedError

    def get_input_channels(self) -> int:
        raise NotImplementedError

    def get_output_channels(self) -> Sequence[int]:
        raise NotImplementedError

    def get_output_scales(self) -> Sequence[int]:
        raise NotImplementedError

class ImageDecoder(_DecoderBase, Registrable):

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                images: Sequence[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def get_output_channels(self) -> int:
        raise NotImplementedError


class BackboneEncoder(ImageEncoder):

    def __init__(self):
        super(BackboneEncoder, self).__init__()
        base = torch_models.resnet34(pretrained=True)
        self.stages = nn.ModuleList()
        self.stages.append(nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool, base.layer1))
        self.stages.append(base.layer2)
        self.stages.append(base.layer3)
        self.stages.append(base.layer4)

    def forward(self, image: torch.Tensor) -> Sequence[torch.Tensor]:
        out = image
        out_images = []
        for stage in self.stages:
            out = stage.forward(out)
            out_images.append(out)
        return out_images

    def _get_test_outputs(self) -> Sequence[torch.Tensor]:
        im = torch.randn(1, self.get_input_channels(), 224, 224)
        outs = []
        out = im
        for stage in self.stages:
            out = stage.forward(out)
            outs.append(out)
        return outs

    def get_output_channels(self) -> Sequence[int]:
        return [x.shape[1] for x in self._get_test_outputs()]

    def get_input_channels(self) -> int:
        return 3

    def get_output_scales(self) -> Sequence[int]:
        outputs = self._get_test_outputs()
        return [outputs[0].shape[-2] / out.shape[-2] for out in outputs]



class Upsample(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, inputs: torch.Tensor):
        out = inputs
        out = self.relu(self.bn(self.conv(out)))
        out = F.interpolate(out, scale_factor=2)
        return out

class FPN(ImageEncoder):
    """
    A feature pyramid network.

    Similar in concept to U-net, it downsamples the image in stages, extracting coarse,
    high-level features. It then upsamples in stages, combining coarse, high-level
    features with fine-grained, low-level features.

    Reference: https://arxiv.org/abs/1612.03144
    """
    def __init__(self,
                 backbone: BackboneEncoder,
                 output_channels: int):
        super(FPN, self).__init__()
        self._backbone = backbone
        self._upscale_factors = [a / b for a, b in zip(self._backbone.get_output_scales()[1:],
                                                       self._backbone.get_output_scales()[:-1])]
        self._output_channels = output_channels
        self.layers = nn.ModuleDict()
        self._convert_layers = nn.ModuleList()
        self._combine_layers = nn.ModuleList()

        for i, n_channels in enumerate(self._backbone.get_output_channels()):
            self._convert_layers.append(StdConv(n_channels, self._output_channels,
                                                       kernel_size=1, stride=1, dropout=0.4,
                                                   padding=0))
            self._combine_layers.append(nn.Sequential(
                SamePad2d(kernel_size=3, stride=1),
                nn.Conv2d(self._output_channels, self._output_channels, kernel_size=3, stride=1)))

    def get_grid_sizes(self, im_h, im_w):
        x = torch.randn(1, 3, im_h, im_w)
        out = self.forward(x)
        return [tuple(o.shape[-2:]) for o in out]

    def forward(self, image: torch.Tensor) -> Sequence[torch.Tensor]:
        outputs = self._backbone.forward(image)

        n = len(outputs)
        converted_images = [self._convert_layers[-1].forward(outputs[-1])]
        out = converted_images[-1]
        for i in range(n - 2, -1, -1):
            out = self._convert_layers[i].forward(outputs[i]) + \
                  F.interpolate(out, scale_factor=self._upscale_factors[i])
            converted_images.append(out)
        combined_images = [layer.forward(im) for layer, im in
                           zip(self._combine_layers, converted_images)]

        return list(reversed(combined_images))

    def get_input_channels(self) -> int:
        return self._backbone.get_input_channels()

    def get_output_channels(self) -> Sequence[int]:
        return [self._output_channels] * len(self._backbone.get_output_channels())

    def get_output_scales(self) -> Sequence[int]:
        return self._backbone.get_output_scales()
