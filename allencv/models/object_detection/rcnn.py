import logging
from typing import Dict, List, Tuple, TypeVar

from overrides import overrides

import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import Average

from allencv.models.object_detection import utils as object_utils
from allencv.modules.im2vec_encoders import Im2VecEncoder, FlattenEncoder

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

TensorList = List[torch.Tensor]


@Model.register("faster_rcnn")
class RCNN(Model):
    """
    A Faster-RCNN model first detects objects in an image using a ``RegionProposalNetwork``.
    Faster-RCNN further separates those objects into classes and refines their bounding boxes.

    Parameters
    ----------
    vocab : ``Vocabulary``
    rpn: ``RPN``
        The region proposal network that detects initial objects.
    roi_feature_extractor: ``Im2VecEncoder``
        Maps each region of interest into a vector of features.
    num_labels: ``int``
        Number of object classe.
    label_namespace: ``str``
        Vocabulary namespace corresponding to labels. By default, we use the "labels" namespace.
    train_rpn: ``bool``
        Whether to include the RPN's loss in the overall loss.
    pooler_resolution: ``int``
        The ROI pooler will output images of this size.
    pooler_sampling_ratio: ``int``
        The sampling ratio for ROIAlign.
    decoder_thresh: ``float``
        The minimum score for an object proposal to make it in the final output.
    decoder_nms_thresh: ``float``
        Proposals that overlap by more than this number will be suppressed into one.
    decoder_detections_per_image: ``int``
        Number of object detections to be proposed per image.
    matcher_high_thresh: ``float``
        Any anchor boxes that overlap with target boxes with an IOU threshold less than
        this number will be considered background.
    matcher_low_thresh: ``float``
        Any anchor boxes that overlap with target boxes with an IOU threshold greater than
        this number will be considered foreground.
    allow_low_quality_matches: ``bool``
        Produce additional matches for predictions that have only low-quality matches.
    batch_size_per_image: ``int``
        The number of examples to include in the loss function for each image in each batch.
    balance_sampling_fraction: ``float``
        What fraction of each batch in the loss computation should be positive
        examples (foreground).
    class_agnostic_bbox_reg: ``bool``
        Whether to use a separate network for each class's bounding box refinement or a single
        network for all classes.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 rpn: Model,
                 roi_box_head: Model,
                 roi_keypoint_head: Model = None,
                 train_rpn: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(RCNN, self).__init__(vocab)
        self._train_rpn = train_rpn
        self.rpn = rpn
        self._box_roi_head = roi_box_head
        self._keypoint_roi_head = roi_keypoint_head

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
        box_out = self._box_roi_head.forward(features, proposals, im_sizes,
                                     class_labels, regression_targets)
        box_out = self._box_roi_head.decode(box_out)

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
                keypoint_proposals = object_utils.unpad(box_out['box_predictions'])
            keypoint_out = self._keypoint_roi_head.forward(features, keypoint_proposals, im_sizes,
                                         keypoint_positions, keypoint_indices)
            for k, v in keypoint_out.items():
                out["keypoint_" + k] = v

        for k, v in box_out.items():
            out["box_" + k] = v

        if boxes is not None:
            rpn_classifier_loss = rpn_out['loss_objectness']
            rpn_regression_loss = rpn_out['loss_rpn_box_reg']
            classifier_loss = box_out['classifier_loss']
            regression_loss = box_out['regression_loss']
            loss = 1. * classifier_loss + regression_loss
            if keypoint_positions is not None:
                loss += 0.1 * out['keypoint_loss']
                self._loss_meters['keypoint_loss'](out['keypoint_loss'].item())
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
            kp_out = self._keypoint_roi_head.decode({k.replace("keypoint_", ""): v for k, v in output_dict.items() if k.startswith("keypoint")})
            output_dict.update({"keypoint_" + k: v for k, v in kp_out.items()})
        output_dict.pop("keypoint_logits")
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {k: v.get_metric(reset) for k, v in self._loss_meters.items()}
        return metrics
