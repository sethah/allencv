from typing import List

import torch

from maskrcnn_benchmark.structures.bounding_box import BoxList


def pad_tensors(tensors: List[torch.Tensor]):
    max_proposals = max([x.shape[0] for x in tensors])
    pad_shape = (len(tensors), max_proposals) + (
    () if tensors[0].dim() <= 1 else tensors[0].shape[1:])
    padded = torch.zeros(pad_shape, device=tensors[0].device)
    for i, t in enumerate(tensors):
        padded[i, :t.shape[0], ...] = t
    return padded


def padded_tensor_to_box_list(padded_tensor: torch.Tensor,
                               image_sizes: torch.Tensor,
                               **kwargs) -> List[BoxList]:
    im_sizes = [(x[1].item(), x[0].item()) for x in image_sizes]
    box_list = []
    for i, (image_boxes, image_size) in enumerate(zip(padded_tensor, im_sizes)):
        # remove boxes that are all zeros (padding)
        mask = torch.any(image_boxes != 0., dim=1)
        bl = BoxList(image_boxes[mask], (image_size[1], image_size[0]))
        for field_name, field_value in kwargs.items():
            bl.add_field(field_name, field_value[i][mask])
        box_list.append(bl)
    return box_list
