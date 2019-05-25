import torch

# apex requirement causes problems when cuda is unavailable
if not torch.cuda.is_available():
    torch.version.cuda = "0.0"
