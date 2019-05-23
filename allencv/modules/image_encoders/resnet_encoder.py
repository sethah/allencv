from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet34, resnet50, resnet18, resnet101, resnet152

from allencv.modules.image_encoders import ImageEncoder
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


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
                 model_str: str,
                 pretrained: bool = False,
                 requires_grad: bool = True) -> None:
        super(ResnetEncoder, self).__init__()
        resnet_model = ResnetEncoder.from_string(model_str, pretrained, requires_grad)
        self._input_channels = 3
        # TODO: frozen batchnorm? no conv bias?
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

    @classmethod
    def from_string(cls, resnet_string: str, pretrained: bool = False, requires_grad: bool = False):
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
        elif resnet_model == 'R50':
            pass
        else:
            raise ValueError(f"Model {resnet_model} is not supported.")

        for param in model.parameters():
            param.requires_grad = requires_grad

        super(PretrainedResnetEncoder, self).__init__(model)



class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = FrozenBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = FrozenBatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = FrozenBatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = FrozenBatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = FrozenBatchNorm2d(planes * self.expansion)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = FrozenBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                FrozenBatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
