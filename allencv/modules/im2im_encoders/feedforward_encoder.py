from overrides import overrides
from typing import Sequence, Union

import torch
import torch.nn as nn

from allencv.modules.im2im_encoders.im2im_encoder import Im2ImEncoder

from allennlp.nn import Activation


class StdConv(nn.Module):

    def __init__(self,
                 nin: int,
                 nout: int,
                 kernel_size: int = 3,
                 activation: Activation = nn.ReLU(),
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 dropout: float = 0.1):
        super(StdConv, self).__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(nout)
        self.drop = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        return self.drop(self.bn(self.activation(self.conv(x))))


@Im2ImEncoder.register("feedforward")
class FeedforwardEncoder(Im2ImEncoder):
    """
    A simple image encoder that passes an input through a series of convolutional layers,
    progressively halving the input size.

    Parameters
    ----------
    input_channels: ``int``
    num_layers: ``int``
        Number of convolutional transforms
    hidden_channels: ``Union[int, Sequence[int]]``
        The number of output channels for each of the convolutional stages.
    activations: ``str``
        The activation function to use after each convolutional stage.
    kernel_sizes: ``Union[int, Sequence[int]]``
        The kernel size for each of the convolutional stages.
    dropout: ``float``
        The amount of dropout for each convolutional stage.
    """

    def __init__(self,
                 input_channels: int,
                 num_layers: int,
                 hidden_channels: Union[int, Sequence[int]],
                 activations: str,
                 kernel_sizes: Union[int, Sequence[int]] = 3,
                 dropout: Union[float, Sequence[float]] = 0.0) -> None:
        super(FeedforwardEncoder, self).__init__()
        if not isinstance(hidden_channels, list):
            hidden_channels = [hidden_channels] * num_layers
        if not isinstance(activations, list):
            activations = [nn.ReLU()] * num_layers
        if not isinstance(dropout, list):
            dropout = [dropout] * num_layers
        if not isinstance(kernel_sizes, list):
            kernel_sizes = [kernel_sizes] * num_layers
        self._activations = activations
        self._kernel_sizes = kernel_sizes
        input_channels = [input_channels] + hidden_channels[:-1]
        conv_layers = []
        for i, (layer_input_channel, layer_output_channel) in enumerate(zip(input_channels, hidden_channels)):
            conv_layers.append(StdConv(layer_input_channel, layer_output_channel,
                                       kernel_size=kernel_sizes[i], activation=activations[i],
                                       dropout=dropout[i]))
            conv_layers.append(StdConv(layer_output_channel, layer_output_channel, stride=2,
                                       kernel_size=kernel_sizes[i],
                                       activation=activations[i]))
        self._conv_layers = nn.ModuleList(conv_layers)
        self._output_channels = hidden_channels[-1]
        self.input_channels = input_channels[0]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        output = inputs
        for layer in self._conv_layers:
            output = layer.forward(output)
        return output

    @overrides
    def get_input_channels(self) -> int:
        return self.input_channels

    @overrides
    def get_output_channels(self) -> int:
        return self._output_channels

