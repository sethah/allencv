import logging
from overrides import overrides
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy

from allencv.common import util
from allencv.modules.image_encoders import ImageEncoder
from allencv.modules.image_decoders import ImageDecoder

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("semantic_segmentation")
class SemanticSegmentationModel(Model):
    """
    This model uses an ``ImageEncoder`` to extract features from an image at one or more
    scales, and then decodes those images to a single feature map, which is then upscaled
    back to the original size. It provides pixel level classifications from a known subset
    possible class labels.
    """
    def __init__(self,
                 encoder: ImageEncoder,
                 decoder: ImageDecoder,
                 num_classes: int,
                 batch_size_per_image: int = None,
                 sample_positive_fraction: float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(SemanticSegmentationModel, self).__init__(None)
        self._batch_size_per_image = batch_size_per_image
        self._sample_positive_fraction = sample_positive_fraction
        # TODO: use vocab to get label namespace?
        self._encoder = encoder
        self.num_classes = num_classes
        self._decoder = decoder
        self.final_conv = nn.Conv2d(in_channels=decoder.get_output_channels(),
                                    out_channels=self.num_classes,
                                    kernel_size=1)
        self._accuracy = CategoricalAccuracy()
        self._loss = nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,  # type: ignore
                image: torch.FloatTensor,
                mask: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ

        encoded_images = self._encoder.forward(image)
        decoded = self._decoder.forward(encoded_images)
        decoded = self.final_conv(decoded)
        # scale image back to label size if possible, otherwise back to original image size
        if mask is not None:
            scale_factor = mask.shape[-2] // decoded.shape[-2]
        else:
            scale_factor = image.shape[-2] // decoded.shape[-2]

        logits = F.interpolate(decoded, scale_factor=scale_factor, mode='bilinear')

        probs = F.softmax(logits, dim=1)

        output_dict = {"logits": logits, "probs": probs}
        if mask is not None:
            flattened_logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
            flattened_mask = mask.view(-1).long()
            # TODO: is it ok to silently filter labels that are out of range?
            ignore = flattened_mask >= self.num_classes
            flattened_mask[ignore] = -1
            # balance the positive and negative samples in each batch
            if self._batch_size_per_image is None:
                batch_size = image.shape[-2] * image.shape[-1]
            else:
                batch_size = self._batch_size_per_image
            neg_idxs, pos_idxs = util.sample_balanced_classes([flattened_mask],
                                                              batch_size * logits.shape[0],
                                                              self._sample_positive_fraction)
            sampled_pos_inds = torch.nonzero(torch.cat(neg_idxs, dim=0)).squeeze(1)
            sampled_neg_inds = torch.nonzero(torch.cat(pos_idxs, dim=0)).squeeze(1)

            sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
            loss = self._loss(flattened_logits[sampled_inds, :], flattened_mask[sampled_inds])
            output_dict["loss"] = loss
            self._accuracy(flattened_logits[~ignore, :], flattened_mask[~ignore])
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output_dict['predicted_mask'] = output_dict['probs'].argmax(dim=1)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {'accuracy': self._accuracy.get_metric(reset)}
        return metrics
