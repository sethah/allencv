import logging
from typing import Dict, List

import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import Average
from overrides import overrides
from torchvision.models.detection.roi_heads import fastrcnn_loss, keypointrcnn_loss
from torchvision.ops import misc as misc_nn_ops

from allencv.models.object_detection import utils as object_utils
from allencv.modules.object_detection.roi_heads import FasterRCNNROIHead, KeypointRCNNROIHead

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

TensorList = List[torch.Tensor]


@Model.register("faster_rcnn2")
class RCNN(Model):
    """
    A Faster-RCNN model first detects objects in an image using a ``RegionProposalNetwork``.
    Faster-RCNN further separates those objects into classes and refines their bounding boxes.

    Parameters
    ----------
    vocab : ``Vocabulary``
    rpn: ``RPN``
        The region proposal network that detects initial objects.
    roi_box_head: ``Model``
        ROI head for bounding boxes.
    roi_keypoint_head: ``Model``
        ROI head for keypoints.
    train_rpn: ``bool``
        Whether to include the RPN's loss in the overall loss.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 rpn: Model,
                 roi_box_head: FasterRCNNROIHead,
                 roi_keypoint_head: KeypointRCNNROIHead = None,
                 num_keypoints: int = None,
                 train_rpn: bool = False,
                 num_labels: int = None,
                 label_namespace: str = "labels",
                 class_agnostic_bbox_reg: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(RCNN, self).__init__(vocab)
        self._train_rpn = train_rpn
        if num_labels is not None:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=label_namespace)
        self.rpn = rpn
        self._box_roi_head = roi_box_head
        self._keypoint_roi_head = roi_keypoint_head
        self._keypoint_upscale = 2

        num_bbox_reg_classes = 2 if class_agnostic_bbox_reg else self._num_labels
        representation_size = roi_box_head.get_output_dim()
        self._box_classifier = nn.Linear(representation_size, self._num_labels)
        self._bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        if self._keypoint_roi_head:
            deconv_kernel = 4
            self._kp_up_scale = 2
            self.kps_score_lowres = misc_nn_ops.ConvTranspose2d(
                self._keypoint_roi_head.get_output_channels(),
                num_keypoints,
                deconv_kernel,
                stride=2,
                padding=deconv_kernel // 2 - 1,
            )
            nn.init.kaiming_normal_(
                self.kps_score_lowres.weight, mode="fan_out", nonlinearity="relu"
            )
            nn.init.constant_(self.kps_score_lowres.bias, 0)

        self._loss_meters = {'roi_cls_loss': Average(), 'roi_reg_loss': Average(),
                             'rpn_cls_loss': Average(), 'rpn_reg_loss': Average()}
        if roi_keypoint_head is not None:
            self._loss_meters['keypoint_loss'] = Average()
        initializer(self)

    def forward(self,
                image: torch.Tensor,  # (batch_size, c, h, w)
                image_sizes: torch.Tensor,  # (batch_size, 2)
                boxes: torch.Tensor = None,  # (batch_size, s, 4)
                box_classes: torch.Tensor = None,
                keypoint_positions: torch.Tensor = None):
        # pylint: disable=arguments-differ
        rpn_out = self.rpn.forward(image, image_sizes, boxes)
        rpn_out = self.rpn.decode(rpn_out)
        features: List[torch.Tensor] = rpn_out['features']

        proposals: TensorList = object_utils.unpad(rpn_out['boxes'])
        if boxes is not None:
            # subsample each set of proposals to a pre-specified number, e.g. 512
            with torch.no_grad():
                # TODO: fix single dimension case?
                gt_labels = [x.squeeze() for x in object_utils.unpad(box_classes)]
                gt_boxes = object_utils.unpad(boxes)
                matched_indices, class_labels, regression_targets = \
                    self._box_roi_head._match_and_sample(proposals, gt_labels, gt_boxes, 0)
        else:
            class_labels = None
            regression_targets = None
            matched_indices = None

        # # roi pool and pass through feature extractor
        # # (b, num_proposal_regions_in_batch, 14, 14)
        im_sizes = [(x[1].item(), x[0].item()) for x in image_sizes]
        out = {'proposals': proposals, 'image_sizes': im_sizes}
        box_features = self._box_roi_head.forward(features, proposals, image_sizes)
        box_regression = self._bbox_pred.forward(box_features)
        class_logits = self._box_classifier.forward(box_features)
        # not necessary unless using keypoint head
        box_proposals, box_scores, box_class_preds = \
            self._box_roi_head.postprocess_detections(class_logits, box_regression,
                                                       proposals, im_sizes)
        loss = 0.
        if self._keypoint_roi_head is not None:
            if keypoint_positions is not None:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                keypoint_indices = []
                for img_id in range(num_images):
                    positive_indices = torch.nonzero(class_labels[img_id] > 0).squeeze(1)
                    keypoint_proposals.append(proposals[img_id][positive_indices])
                    keypoint_indices.append(matched_indices[img_id][positive_indices])
            else:
                keypoint_indices = None
                keypoint_proposals = object_utils.unpad(box_proposals)
            keypoint_features = self._keypoint_roi_head.forward(features, keypoint_proposals,
                                                                im_sizes)
            keypoint_logits = self.kps_score_lowres.forward(keypoint_features)
            keypoint_logits = misc_nn_ops.interpolate(
                keypoint_logits, scale_factor=self._keypoint_upscale, mode="bilinear",
                align_corners=False)
            out['keypoint_logits'] = keypoint_logits
            if keypoint_positions is not None:
                keypoint_loss = keypointrcnn_loss(keypoint_logits, keypoint_proposals,
                    keypoint_positions, keypoint_indices)
                out['keypoint_loss'] = keypoint_loss
                loss += keypoint_loss
                self._loss_meters['keypoint_loss'](out['keypoint_loss'].item())

        out['box_proposals'] = box_proposals
        out['box_scores'] = box_scores
        out['box_labels'] = box_class_preds

        if boxes is not None:
            rpn_classifier_loss = rpn_out['loss_objectness']
            rpn_regression_loss = rpn_out['loss_rpn_box_reg']
            classifier_loss, regression_loss = fastrcnn_loss(
                class_logits, box_regression, class_labels, regression_targets)
            loss += 1. * classifier_loss + regression_loss
            if self._train_rpn:
                loss += rpn_classifier_loss + rpn_regression_loss
            out['loss'] = loss
            self._loss_meters['rpn_cls_loss'](rpn_classifier_loss.item())
            self._loss_meters['rpn_reg_loss'](rpn_regression_loss.item())
            self._loss_meters['roi_cls_loss'](classifier_loss.item())
            self._loss_meters['roi_reg_loss'](regression_loss.item())
        return out

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output_dict.pop("proposals")
        if self._keypoint_roi_head is not None:
            kp_logits = output_dict['keypoint_logits']
            box_proposals = output_dict['box_proposals']
            kp_proposals, kp_scores = self._keypoint_roi_head.postprocess_detections(kp_logits,
                                                                                     box_proposals)
            output_dict['keypoint_proposals'] = kp_proposals
            output_dict['keypoint_scores'] = kp_scores
        all_predictions: List[torch.Tensor] = output_dict['box_labels']
        if self.vocab is not None:
            idx2token = self.vocab.get_index_to_token_vocabulary(namespace="labels")
            all_classes = []
            for predictions in all_predictions:
                all_classes.append([idx2token.get(x, 'background') for x in predictions.cpu().numpy().tolist()])
            output_dict['box_class'] = all_classes
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {k: v.get_metric(reset) for k, v in self._loss_meters.items()}
        return metrics
