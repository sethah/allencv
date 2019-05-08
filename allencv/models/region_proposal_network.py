import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import Average

from allencv.modules.image_encoders import ImageEncoder

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import ImageList
from maskrcnn_benchmark.modeling.rpn.loss import RPNLossComputation, generate_rpn_labels
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from maskrcnn_benchmark.modeling.rpn.anchor_generator import AnchorGenerator
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.rpn.inference import RPNPostProcessor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("rpn")
class RPN(Model):
    """
    A region proposal network
    """

    def __init__(self,
                 backbone: ImageEncoder,
                 positive_fraction: float = 0.5,
                 match_thresh_low: float = 0.3,
                 match_thresh_high: float = 0.7,
                 anchor_sizes: List[int] = [128, 256, 512],
                 anchor_aspect_ratios: List[float] = [0.5, 1.0, 2.0],
                 anchor_strides: List[int] = [8, 16, 32],
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(RPN, self).__init__(None)

        # the BoxCoder just converts the relative regression offsets into absolute
        # coordinates
        box_coder = BoxCoder(weights=(1., 1., 1., 1.))
        batch_size_per_image = 256

        # sampler is responsible for selecting a subset of anchor boxes for computing the loss
        # this makes sure each batch has reasonable balance of foreground/background labels
        # it selects `batch_size_per_image` total boxes
        sampler = BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)

        # matcher decides if an anchor box is a foreground or background based on how much
        # it overlaps with the nearest target box
        matcher = Matcher(match_thresh_high, match_thresh_low, allow_low_quality_matches=False)

        # the decoder will choose the highest scoring foreground anchor boxes and then perform
        # non-max suppression on those to eliminate duplicates. After that it will choose
        # a further subset of the ones that survived nms as proposals from the RPN
        self.decoder = RPNPostProcessor(pre_nms_top_n=6000,
                                        post_nms_top_n=300,
                                        nms_thresh=0.7,
                                        min_size=0,
                                        box_coder=box_coder,
                                        fpn_post_nms_top_n=2000,
                                        fpn_post_nms_per_batch=True)
        self.backbone = backbone
        self.loss_evaluator = RPNLossComputation(matcher,
                                                 sampler,
                                                 box_coder,
                                                 generate_rpn_labels)
        self.anchor_generator = AnchorGenerator(anchor_sizes, anchor_aspect_ratios, anchor_strides,
                                                straddle_thresh=200)
        self.num_anchors = self.anchor_generator.num_anchors_per_location()[0]

        # TODO: backbone must produce maps with all same number of channels
        in_channels = backbone.get_output_channels()[0]
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(in_channels, self.num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, self.num_anchors * 4, kernel_size=1, stride=1
        )
        # initializer(self)

        # TODO: use initializer?
        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)
        self._loss_meters = {'rpn_cls_loss': Average(), 'rpn_reg_loss': Average()}

    def forward(self,
                image: torch.Tensor,  # (batch_size, c, h, w)
                image_sizes: torch.Tensor,  # (batch_size, 2)
                boxes: torch.Tensor = None,  # (batch_size, max_boxes_in_batch, 4)
                box_classes: torch.Tensor = None
                ) -> Dict[str, torch.Tensor]:
        features = self.backbone.forward(image)
        objectness: List[torch.Tensor] = []
        rpn_box_regression: List[torch.Tensor] = []
        for feature in features:
            t = F.relu(self.conv(feature))
            objectness.append(self.cls_logits(t))
            rpn_box_regression.append(self.bbox_pred(t))
        im_sizes = [(x[1].item(), x[0].item()) for x in image_sizes]
        image_list = ImageList(image, im_sizes)
        anchors: List[List[BoxList]] = self.anchor_generator(image_list, features)

        # you get a list of proposal boxes for each image in the batch
        proposals: List[BoxList] = self.decoder(anchors, objectness, rpn_box_regression)
        proposal_scores = [b.get_field("objectness").unsqueeze(1) for b in proposals]
        proposals: torch.Tensor = self._pad_tensors([p.bbox for p in proposals])
        proposal_scores: torch.Tensor = self._pad_tensors(proposal_scores)[:, :, 0]

        out = {'features': features, 'proposals': proposals,
               'proposal_scores': proposal_scores}
        if boxes is not None:
            box_list: List[BoxList] = self._padded_tensor_to_box_list(boxes, image_sizes)
            loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
                anchors, objectness, rpn_box_regression, box_list
            )
            out["loss_objectness"] = loss_objectness
            out["loss_rpn_box_reg"] = loss_rpn_box_reg
            self._loss_meters['rpn_cls_loss'](loss_objectness.item())
            self._loss_meters['rpn_reg_loss'](loss_rpn_box_reg.item())
            out["loss"] = loss_objectness + loss_rpn_box_reg
        return out

    def _pad_tensors(self, tensors: List[torch.Tensor]):
        max_proposals = max([x.shape[0] for x in tensors])
        padded = torch.zeros(len(tensors), max_proposals, tensors[0].shape[1],
                             device=tensors[0].device)
        for i, t in enumerate(tensors):
            padded[i, :t.shape[0], :] = t
        return padded

    def _padded_tensor_to_box_list(self,
                                   padded_tensor: torch.Tensor,
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

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {k: v.get_metric(reset) for k, v in self._loss_meters.items()}
        return metrics


