import torch

# apex requirement causes problems when cuda is unavailable
# TODO: this is not cool in general to do.
if not torch.cuda.is_available():
    torch.version.cuda = "9.0"
