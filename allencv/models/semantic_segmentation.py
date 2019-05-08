from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy

from allencv.modules.image_encoders import ImageEncoder
from allencv.modules.image_decoders import ImageDecoder


@Model.register("semantic_segmentation")
class SemanticSegmentationModel(Model):
    """
    """
    def __init__(self,
                 encoder: ImageEncoder,
                 decoder: ImageDecoder,
                 num_classes: int,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(SemanticSegmentationModel, self).__init__(None)
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
            loss = self._loss(logits, mask.squeeze().long())
            output_dict["loss"] = loss
            self._accuracy(logits.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes), label.view(-1))

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {'accuracy': self._accuracy.get_metric(reset)}
        return metrics
