import torch


class _EncoderBase(torch.nn.Module):
    """
    Base class for image encoders.
    """
    def __init__(self) -> None:
        super(_EncoderBase, self).__init__()
