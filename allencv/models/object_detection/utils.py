from typing import List

import torch


def pad_tensors(tensors: List[torch.Tensor]):
    max_proposals = max([x.shape[0] for x in tensors])
    pad_shape = (len(tensors), max_proposals) + (
    () if tensors[0].dim() <= 1 else tensors[0].shape[1:])
    padded = torch.zeros(pad_shape, device=tensors[0].device)
    for i, t in enumerate(tensors):
        padded[i, :t.shape[0], ...] = t
    return padded


def padded_tensor_to_tensor_list(padded_tensor: torch.Tensor) -> List[torch.Tensor]:
    box_list = []
    for image_boxes in padded_tensor:
        # remove boxes that are all zeros (padding)
        mask = torch.any(image_boxes != 0., dim=-1)
        box_list.append(image_boxes[mask])
    return box_list

def unpad(padded_tensor: torch.Tensor) -> List[torch.Tensor]:
    box_list = []
    for image_boxes in padded_tensor:
        # remove boxes that are all zeros (padding)
        mask = torch.any(image_boxes != 0., dim=-1)
        box_list.append(image_boxes[mask])
    return box_list
