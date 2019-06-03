from typing import Dict, List, Tuple

from collections import OrderedDict
import logging
from overrides import overrides

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models.model import Model

from torchvision.models.detection import _utils as det_utils
from torchvision.ops import boxes as box_ops
from torchvision.ops.poolers import MultiScaleRoIAlign
from torchvision.models.detection.roi_heads import fastrcnn_loss, keypointrcnn_loss, keypointrcnn_inference
from torchvision.ops import misc as misc_nn_ops

from allencv.models.object_detection import utils as object_utils
from allencv.modules.im2vec_encoders import Im2VecEncoder
from allencv.modules.im2im_encoders import Im2ImEncoder

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

TensorList = List[torch.Tensor]


@Model.register("faster_rcnn_roi_box")
class FasterRCNNROIHead(Model):
    """
    A Faster-RCNN model first detects objects in an image using a ``RegionProposalNetwork``.
    Faster-RCNN further separates those objects into classes and refines their bounding boxes.

    Parameters
    ----------
    vocab : ``Vocabulary``
    feature_extractor: ``Im2VecEncoder``
        Maps each region of interest into a vector of features.
    num_labels: ``int``
        Number of object classe.
    label_namespace: ``str``
        Vocabulary namespace corresponding to labels. By default, we use the "labels" namespace.
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
                 feature_extractor: Im2VecEncoder,
                 pooler_resolution: int = 7,
                 pooler_sampling_ratio: int = 2,
                 num_labels: int = None,
                 label_namespace: str = "labels",
                 decoder_thresh: float = 0.1,
                 decoder_nms_thresh: float = 0.5,
                 decoder_detections_per_image: int = 100,
                 matcher_high_thresh: float = 0.5,
                 matcher_low_thresh: float = 0.5,
                 allow_low_quality_matches: bool = True,
                 batch_size_per_image: int = 256,
                 balance_sampling_fraction: float = 0.25,
                 class_agnostic_bbox_reg: bool = False):
        super(FasterRCNNROIHead, self).__init__(vocab)
        if num_labels is not None:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=label_namespace)
        self.roi_pool = MultiScaleRoIAlign(
            featmap_names=[0, 1, 2, 3],
            output_size=pooler_resolution,
            sampling_ratio=pooler_sampling_ratio)
        self.feature_extractor = feature_extractor
        num_bbox_reg_classes = 2 if class_agnostic_bbox_reg else self._num_labels
        representation_size = self.feature_extractor.get_output_dim()
        self.cls_score = nn.Linear(representation_size, self._num_labels)
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)
        self.box_coder = det_utils.BoxCoder(weights=(10., 10., 5., 5.))
        self.decoder_thresh = decoder_thresh
        self.decoder_nms_thresh = decoder_nms_thresh
        self.decoder_detections_per_image = decoder_detections_per_image
        self.proposal_matcher = det_utils.Matcher(
            high_threshold=matcher_high_thresh,
            low_threshold=matcher_low_thresh,
            allow_low_quality_matches=allow_low_quality_matches)
        self.sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction=balance_sampling_fraction)

    def forward(self,
                features: List[torch.Tensor],
                proposals: List[torch.Tensor],
                image_shapes: List[Tuple[int, int]],
                class_labels: List[torch.Tensor] = None,
                regression_targets: List[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        _features = OrderedDict([(i, f) for i, f in enumerate(features)])
        pooled = self.roi_pool(_features, proposals, image_shapes)
        box_features = self.feature_extractor.forward(pooled)
        class_logits = self.cls_score(box_features)
        box_regression = self.bbox_pred(box_features)
        out = {'regression': box_regression, 'class_logits': class_logits,
               'proposals': proposals, 'image_sizes': image_shapes}
        if class_labels is not None:
            classifier_loss, regression_loss = fastrcnn_loss(
                class_logits, box_regression, class_labels, regression_targets)
            out.update({'classifier_loss': classifier_loss, 'regression_loss': regression_loss})
        return out

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # TODO: provide split class and regression tensors?
        boxes, scores, labels = self._postprocess_detections(
            output_dict['class_logits'],
            output_dict['regression'],
            output_dict['proposals'],
            output_dict['image_sizes'])
        output_dict = {}
        output_dict['box_scores'] = object_utils.pad_tensors(scores)
        output_dict['class_predictions'] = object_utils.pad_tensors(labels)
        output_dict['box_predictions'] = object_utils.pad_tensors(boxes)
        all_predictions = output_dict['class_predictions']
        all_predictions = all_predictions.cpu().data.numpy()
        if all_predictions.ndim == 2:
            predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
        else:
            predictions_list = [all_predictions]
        if self.vocab is not None:
            idx2token = self.vocab.get_index_to_token_vocabulary(namespace="labels")
            all_classes = []
            for predictions in predictions_list:
                all_classes.append([idx2token.get(x, 'background') for x in predictions])
            output_dict['class'] = all_classes
        return output_dict

    def _match_and_sample(self,
                          proposals: TensorList,
                          class_labels: TensorList,
                          box_labels: TensorList,
                          background_label: int = 0) -> Tuple[TensorList, TensorList, TensorList]:

        # figure out which proposals correspond to actual labels
        # matched_idxs = [0, 0, 1, 2, 2, 0, 2, 1, 1] if there were three ground truth boxes
        matched_idxs, labels = self._assign_targets_to_proposals(
            proposals, box_labels, class_labels, background_label)
        # subsample
        sampled_pos_inds, sampled_neg_inds = self.sampler(labels)
        sampled_inds = []
        for pos_inds_img, neg_inds_img in zip(sampled_pos_inds, sampled_neg_inds):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            sampled_inds.append(img_sampled_inds)

        # sample the labels and proposals down so we have somewhat balanced classes
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            matched_gt_boxes.append(box_labels[img_id][matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return matched_idxs, labels, regression_targets

    def _assign_targets_to_proposals(self,
                                     proposals: TensorList,
                                     gt_boxes: TensorList,
                                     gt_labels: TensorList,
                                     background_label: int = 0) -> Tuple[TensorList, TensorList]:
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
            matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

            clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

            labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
            labels_in_image = labels_in_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_in_image[bg_inds] = background_label

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def _postprocess_detections(self,
                                class_logits: torch.Tensor,
                                box_regression: torch.Tensor,
                                proposals: TensorList,
                                image_shapes: List[Tuple[int, int]]):
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes, pred_scores, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.decoder_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.decoder_nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.decoder_detections_per_image]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels


@Model.register("faster_rcnn_roi_keypoint")
class KeypointRCNNROIHead(Model):

    def __init__(self,
                 feature_extractor: Im2ImEncoder,
                 num_keypoints: int,
                 pooler_resolution: int = 14,
                 pooler_sampling_ratio: int = 2):
        super(KeypointRCNNROIHead, self).__init__(None)
        self.feature_extractor = feature_extractor

        input_features = feature_extractor.get_output_channels()
        deconv_kernel = 4
        self.kps_score_lowres = misc_nn_ops.ConvTranspose2d(
            input_features,
            num_keypoints,
            deconv_kernel,
            stride=2,
            padding=deconv_kernel // 2 - 1,
        )
        nn.init.kaiming_normal_(
            self.kps_score_lowres.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.kps_score_lowres.bias, 0)
        self.up_scale = 2
        self.out_channels = num_keypoints

        self.roi_pool = MultiScaleRoIAlign(
            featmap_names=[0, 1, 2, 3],
            output_size=pooler_resolution,
            sampling_ratio=pooler_sampling_ratio)

    def forward(self,
                features: List[torch.Tensor],
                proposals: List[torch.Tensor],
                image_shapes: List[Tuple[int, int]],
                keypoint_positions: List[torch.Tensor],
                keypoint_indices: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        _features = OrderedDict([(i, f) for i, f in enumerate(features)])
        keypoint_features = self.roi_pool(_features, proposals, image_shapes)
        keypoint_features = self.feature_extractor.forward(keypoint_features)
        # (N x K x H x W)
        keypoint_logits = self.kps_score_lowres(keypoint_features)
        keypoint_logits = misc_nn_ops.interpolate(
            keypoint_logits, scale_factor=self.up_scale, mode="bilinear", align_corners=False
        )
        out = {'logits': keypoint_logits, 'proposals': proposals}
        if keypoint_positions is not None:
            keypoint_loss = keypointrcnn_loss(
                keypoint_logits, proposals,
                keypoint_positions, keypoint_indices)
            out.update({'loss': keypoint_loss})
        return out

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        keypoints_probs, kp_scores = keypointrcnn_inference(output_dict['logits'],
                                                            output_dict['proposals'])
        proposals = []
        scores = []
        for keypoint_prob, kps in zip(keypoints_probs, kp_scores):
            proposals.append(keypoint_prob)
            scores.append(kps)
        return {'proposals': proposals, 'scores': scores}
