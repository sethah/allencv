import torch


class _DecoderBase(torch.nn.Module):
    """
    Base class for image decoders.
    """
    def __init__(self) -> None:
        super(_DecoderBase, self).__init__()