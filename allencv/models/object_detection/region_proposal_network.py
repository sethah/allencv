from typing import Dict, List, Tuple
import logging
from overrides import overrides

import torch
import torch.nn.functional as F
from torchvision.models.detection.rpn import AnchorGenerator, concat_box_prediction_layers
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection import _utils as det_utils
from torchvision.ops import boxes as box_ops
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import Average

from allencv.nn.common import RPNHead
from allencv.modules.image_encoders import ImageEncoder, ResnetEncoder, FPN
from allencv.models.object_detection import utils as object_utils

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("rpn")
class RPN(Model):
    """
    A region proposal network identifies regions of an input image that correspond
    to objects of various classes. It is typically used as the first stage in a
    Faster R-CNN.

    Parameters
    ----------
    backbone: ``ImageEncoder``
        Encoder network that provides features from the input image at various scales.
    positive_fraction: ``float``
        What fraction of each batch in the loss computation should be positive
        examples (foreground).
    match_thresh_low: ``float``
        Any anchor boxes that overlap with target boxes with an IOU threshold less than
        this number will be considered background.
    match_thresh_high: ``float``
        Any anchor boxes that overlap with target boxes with an IOU threshold greater than
        this number will be considered foreground.
    anchor_sizes: ``List[int]``
        Size of anchor boxes at various levels.
    anchor_aspect_ratios: ``List[float]``
        Which aspect ratios to include as anchors for each anchor location.
    batch_size_per_image: ``int``
        The number of examples to include in the loss function for each image in each batch.
    pre_nms_top_n: ``int``
        The number of proposals to keep before non-maximum suppression is applied.
    post_nms_top_n: ``int``
        The number of proposals after non-maximum suppression is applied.
    nms_thresh: ``float``
        Proposal boxes that overlap by more than this amount will be suppressed.
    min_size: ``int``
        The minimum size of a proposal bounding box.
    fpn_post_nms_top_n: ``int``
        How many proposals to keep after non-max supprossion is applied.
    fpn_post_nms_per_batch: ``bool``
        Whether the `fpn_post_nms_top_n` corresponds to each batch or each image.
    allow_low_quality_matches: ``bool``
        Produce additional matches for predictions that have only low-quality matches.
    """

    def __init__(self,
                 backbone: ImageEncoder,
                 positive_fraction: float = 0.5,
                 match_thresh_low: float = 0.3,
                 match_thresh_high: float = 0.7,
                 anchor_sizes: List[int] = (128, 256, 512),
                 anchor_aspect_ratios: List[float] = (0.5, 1.0, 2.0),
                 batch_size_per_image: int = 256,
                 pre_nms_top_n: int = 6000,
                 post_nms_top_n: int = 300,
                 nms_thresh: int = 0.7,
                 min_size: int = 0,
                 fpn_post_nms_top_n: int = 1000,
                 fpn_post_nms_per_batch: int = True,
                 allow_low_quality_matches: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(RPN, self).__init__(None)
        self._rpn_head = RPNHead(256, 3)
        self.min_size = min_size
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.post_nms_top_n = post_nms_top_n

        # the BoxCoder just converts the relative regression offsets into absolute
        # coordinates
        self.box_coder = det_utils.BoxCoder(weights=(1., 1., 1., 1.))

        # sampler is responsible for selecting a subset of anchor boxes for computing the loss
        # this makes sure each batch has reasonable balance of foreground/background labels
        # it selects `batch_size_per_image` total boxes
        self.sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image,
                                                                 positive_fraction)

        # matcher decides if an anchor box is a foreground or background based on how much
        # it overlaps with the nearest target box
        self.proposal_matcher = det_utils.Matcher(
                match_thresh_high,
                match_thresh_low,
                allow_low_quality_matches=allow_low_quality_matches)

        self.backbone = backbone
        self.anchor_generator = AnchorGenerator(anchor_sizes, anchor_aspect_ratios)
        self.num_anchors = self.anchor_generator.num_anchors_per_location()[0]

        # initializer(self)

        # TODO: use initializer?
        # for l in [self.conv, self.cls_logits, self.bbox_pred]:
        #     torch.nn.init.normal_(l.weight, std=0.01)
        #     torch.nn.init.constant_(l.bias, 0)
        self._loss_meters = {'rpn_cls_loss': Average(), 'rpn_reg_loss': Average()}

    def assign_targets_to_anchors(self,
                                  anchors: List[torch.Tensor],
                                  targets: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        labels = []
        matched_gt_boxes = []
        for anchors_per_image, gt_boxes in zip(anchors, targets):
            match_quality_matrix = box_ops.box_iou(gt_boxes, anchors_per_image)
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            # get the targets corresponding GT for each proposal
            # NB: need to clamp the indices because we can have a single
            # GT in the image, and matched_idxs can be -2, which goes
            # out of bounds
            matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        result = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            result.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(result, dim=1)

    def forward(self,
                image: torch.Tensor,  # (batch_size, c, h, w)
                image_sizes: torch.Tensor,  # (batch_size, 2)
                boxes: torch.Tensor = None,  # (batch_size, max_boxes_in_batch, 4)
                box_classes: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        im_sizes = [(x[1].item(), x[0].item()) for x in image_sizes]
        image_list = ImageList(image, im_sizes)
        features = self.backbone.forward(image)
        objectness, rpn_box_regression = self._rpn_head(features)
        anchors: List[torch.Tensor] = self.anchor_generator(image_list, features)
        num_anchors_per_level = [o[0].numel() for o in objectness]
        objectness, rpn_box_regression = \
            concat_box_prediction_layers(objectness, rpn_box_regression)

        out = {'features': features,
               'objectness': objectness,
               'rpn_box_regression': rpn_box_regression,
               'anchors': anchors,
               'sizes': image_sizes,
               'num_anchors_per_level': num_anchors_per_level}
        if boxes is not None:
            labels, matched_gt_boxes = self.assign_targets_to_anchors(
                    anchors, object_utils.padded_tensor_to_tensor_list(boxes))
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)

            sampled_pos_inds, sampled_neg_inds = self.sampler(labels)
            sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
            sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

            sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

            objectness = objectness.flatten()

            labels = torch.cat(labels, dim=0)
            regression_targets = torch.cat(regression_targets, dim=0)

            loss_rpn_box_reg = F.l1_loss(
                    rpn_box_regression[sampled_pos_inds],
                    regression_targets[sampled_pos_inds],
                    reduction="sum",
            ) / (sampled_inds.numel())

            loss_objectness = F.binary_cross_entropy_with_logits(
                    objectness[sampled_inds], labels[sampled_inds]
            )
            self._loss_meters['rpn_cls_loss'](loss_objectness.item())
            self._loss_meters['rpn_reg_loss'](loss_rpn_box_reg.item())
            out["loss_objectness"] = loss_objectness
            out["loss_rpn_box_reg"] = loss_rpn_box_reg
            out["loss"] = loss_objectness + 10*loss_rpn_box_reg
        return out

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        num_images = proposals.shape[0]
        device = proposals.device
        # do not backprop throught objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        levels = [
                torch.full((n,), idx, dtype=torch.int64, device=device)
                for idx, n in enumerate(num_anchors_per_level)
        ]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)
        batch_idx = torch.arange(num_images, device=device)[:, None]
        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape in zip(proposals, objectness, levels, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.post_nms_top_n]
            boxes, scores = boxes[keep], scores[keep]
            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # you get a list of proposal boxes for each image in the batch
        proposals = self.box_coder.decode(
                output_dict['rpn_box_regression'], output_dict['anchors'])
        num_images = len(output_dict['anchors'])
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals,
                                              output_dict['objectness'],
                                              output_dict['sizes'],
                                              output_dict['num_anchors_per_level'])
        output_dict['proposals'] = boxes
        output_dict['proposal_scores'] = scores
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {k: v.get_metric(reset) for k, v in self._loss_meters.items()}
        return metrics


# @Model.register("pretrained")
# class PretrainedRPN(RPN):
#
#     @classmethod
#     def from_params(cls, params: Params):
#         model = load_archive(params.pop("archive_file")).model
#         requires_grad = params.pop("requires_grad")
#         for p in model.parameters():
#             p.requires_grad = requires_grad
#         return model
#
#
@Model.register("detectron_rpn")
class PretrainedDetectronRPN(RPN):

    def __init__(self,
                 anchor_sizes: List[int] = None,
                 anchor_aspect_ratios: List[float] = None,
                 batch_size_per_image: int = 256):
        backbone = ResnetEncoder('resnet50')
        fpn_backbone = FPN(backbone, 256)
        if anchor_sizes is None:
            anchor_sizes = [[32], [64], [128], [256], [512]]
        if anchor_aspect_ratios is None:
            anchor_aspect_ratios = [[0.5, 1.0, 2.0], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0],
                              [0.5, 1.0, 2.0], [0.5, 1.0, 2.0]]
        super(PretrainedDetectronRPN, self).__init__(fpn_backbone,
                                                     anchor_aspect_ratios=anchor_aspect_ratios,
                                                     anchor_sizes=anchor_sizes,
                                                     batch_size_per_image=batch_size_per_image)
        frcnn = fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
        backbone.load_state_dict(frcnn.backbone.body.state_dict())
        self._rpn_head.load_state_dict(frcnn.rpn.head.state_dict())

        # pylint: disable = protected-access
        fpn_backbone._convert_layers.load_state_dict(frcnn.backbone.fpn.inner_blocks.state_dict())
        fpn_backbone._combine_layers.load_state_dict(frcnn.backbone.fpn.layer_blocks.state_dict())
