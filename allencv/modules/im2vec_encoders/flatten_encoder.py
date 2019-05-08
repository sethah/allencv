import torch

from allennlp.modules import FeedForward

from allencv.modules.im2vec_encoders.im2vec_encoder import Im2VecEncoder


@Im2VecEncoder.register("flatten")
class FlattenEncoder(Im2VecEncoder):
    """
    An ``Im2VecEncoder`` that simply flattens the input image and optionally passes it
    through a feedforward network.
    """

    def __init__(self,
                 input_channels: int,
                 input_height: int,
                 input_width: int,
                 feedforward: FeedForward = None) -> None:
        super(FlattenEncoder, self).__init__()
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.feedforward = feedforward

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs.view(inputs.shape[0], -1)
        if self.feedforward is not None:
            output = self.feedforward(output)
        return output

    def get_input_channels(self) -> int:
        return self.input_channels

    def get_output_dim(self) -> int:
        return self.input_height * self.input_width * self.input_channels \
            if self.feedforward is None else self.feedforward.get_output_dim()
