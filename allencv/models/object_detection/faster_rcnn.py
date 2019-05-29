import logging
from overrides import overrides
from typing import Dict, List, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import Average

from torchvision.models.detection import _utils as det_utils
from torchvision.ops import boxes as box_ops
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.ops.poolers import MultiScaleRoIAlign
from torchvision.models.detection.roi_heads import fastrcnn_loss

from allencv.models.object_detection.region_proposal_network import RPN
from allencv.models.object_detection import utils as object_utils
from allencv.modules.im2vec_encoders import Im2VecEncoder, FlattenEncoder

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("faster_rcnn")
class FasterRCNN(Model):
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
                 roi_feature_extractor: Im2VecEncoder,
                 num_labels: int = None,
                 label_namespace: str = "labels",
                 train_rpn: bool = False,
                 pooler_resolution: int = 7,
                 pooler_sampling_ratio: int = 2,
                 decoder_thresh: float = 0.1,
                 decoder_nms_thresh: float = 0.5,
                 decoder_detections_per_image: int = 100,
                 matcher_high_thresh: float = 0.5,
                 matcher_low_thresh: float = 0.5,
                 allow_low_quality_matches: bool = True,
                 batch_size_per_image: int = 64,
                 balance_sampling_fraction: float = 0.25,
                 class_agnostic_bbox_reg: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(FasterRCNN, self).__init__(vocab)
        self.decoder_thresh = decoder_thresh
        self.decoder_nms_thresh = decoder_nms_thresh
        self.decoder_detections_per_image = decoder_detections_per_image
        self._train_rpn = train_rpn
        self.rpn = rpn
        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=label_namespace)
        self.feature_extractor = roi_feature_extractor
        num_bbox_reg_classes = 2 if class_agnostic_bbox_reg else self._num_labels
        representation_size = roi_feature_extractor.get_output_dim()
        self.cls_score = nn.Linear(representation_size, self._num_labels)
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        self.proposal_matcher = det_utils.Matcher(
            high_threshold=matcher_high_thresh,
            low_threshold=matcher_low_thresh,
            allow_low_quality_matches=allow_low_quality_matches)
        self.sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image,
                                                  positive_fraction=balance_sampling_fraction)
        self.box_coder = det_utils.BoxCoder(weights=(10., 10., 5., 5.))
        self.box_roi_pool = MultiScaleRoIAlign(
                featmap_names=[0, 1, 2, 3],
                output_size=pooler_resolution,
                sampling_ratio=pooler_sampling_ratio)
        self._loss_meters = {'roi_cls_loss': Average(), 'roi_reg_loss': Average(),
                             'rpn_cls_loss': Average(), 'rpn_reg_loss': Average()}
        initializer(self)

    def forward(self,
                image: torch.Tensor,  # (batch_size, c, h, w)
                image_sizes: torch.Tensor,  # (batch_size, 2)
                boxes: torch.Tensor = None,  # (batch_size, s, 4)
                box_classes: torch.Tensor = None):
        rpn_out = self.rpn.forward(image, image_sizes, boxes)
        rpn_out = self.rpn.decode(rpn_out)
        features: List[torch.Tensor] = rpn_out['features']

        proposals: List[torch.Tensor] = \
            object_utils.padded_tensor_to_tensor_list(rpn_out['proposals'])
        if boxes is not None:
            # subsample each set of proposals to a pre-specified number, e.g. 512
            with torch.no_grad():
                # TODO: fix single dimension case?
                gt_labels = [x.squeeze() for x in object_utils.padded_tensor_to_tensor_list(box_classes)]
                gt_boxes = object_utils.padded_tensor_to_tensor_list(boxes)
                matched_idxs, labels = self._assign_targets_to_proposals(proposals, gt_boxes,
                                                                        gt_labels)
                # subsample
                sampled_pos_inds, sampled_neg_inds = self.sampler(labels)
                sampled_inds = []
                for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
                        zip(sampled_pos_inds, sampled_neg_inds)
                ):
                    img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
                    sampled_inds.append(img_sampled_inds)

                matched_gt_boxes = []
                num_images = len(proposals)
                for img_id in range(num_images):
                    img_sampled_inds = sampled_inds[img_id]
                    proposals[img_id] = proposals[img_id][img_sampled_inds]
                    labels[img_id] = labels[img_id][img_sampled_inds]
                    matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
                    matched_gt_boxes.append(gt_boxes[img_id][matched_idxs[img_id]])

                regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)

        # roi pool and pass through feature extractor
        # (b, num_proposal_regions_in_batch, 14, 14)
        im_sizes = [(x[1].item(), x[0].item()) for x in image_sizes]
        _features = OrderedDict([(i, f) for i, f in enumerate(features)])
        pooled = self.box_roi_pool(_features, proposals, im_sizes)
        box_features = self.feature_extractor.forward(pooled)
        class_logits = self.cls_score(box_features)
        box_regression = self.bbox_pred(box_features)

        # pooler, combines logits from all images in the batch to one large batch
        # you would need to split the tensor by the number of proposals in each image to
        # recover the per-image logits
        out = {'class_logits': class_logits,
               'box_regression': box_regression,
               'proposals': proposals,
               'image_sizes': im_sizes}

        if boxes is not None:
            rpn_classifier_loss = rpn_out['loss_objectness']
            rpn_regression_loss = rpn_out['loss_rpn_box_reg']
            classifier_loss, regression_loss = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            loss = 0.003 * classifier_loss + regression_loss
            if self._train_rpn:
                loss += 0.1 * rpn_classifier_loss + rpn_regression_loss
            out['loss'] = loss
            self._loss_meters['rpn_cls_loss'](rpn_out['loss_objectness'].item())
            self._loss_meters['rpn_reg_loss'](rpn_out['loss_rpn_box_reg'].item())
            self._loss_meters['roi_cls_loss'](classifier_loss.item())
            self._loss_meters['roi_reg_loss'](regression_loss.item())
        return out

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # TODO: provide split class and regression tensors?
        boxes, scores, labels = self._postprocess_detections(output_dict['class_logits'],
                                                            output_dict['box_regression'],
                                                            output_dict['proposals'],
                                                            output_dict['image_sizes'])
        output_dict['scores'] = object_utils.pad_tensors(scores)
        output_dict['labels'] = object_utils.pad_tensors(labels)
        output_dict['decoded'] = object_utils.pad_tensors(boxes)
        output_dict.pop("proposals")

        # convert label indices to classes
        all_predictions = output_dict['labels']
        all_predictions = all_predictions.cpu().data.numpy()
        if all_predictions.ndim == 2:
            predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
        else:
            predictions_list = [all_predictions]
        if self.vocab is not None:
            idx2token = self.vocab.get_index_to_token_vocabulary(namespace="labels")
            all_classes = []
            for predictions in predictions_list:
                all_classes.append([idx2token[x] for x in predictions])
            output_dict['class'] = all_classes
        # split_logits = class_logits.split([len(p) for p in sampled_proposals])
        # # [(13, 4), (17, 4)] -> (2, 17, 4)
        # split_logits = self.rpn._pad_tensors(split_logits)
        # split_box_regression = box_regression.split([len(p) for p in sampled_proposals])
        # split_box_regression = self.rpn._pad_tensors(split_box_regression)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {k: v.get_metric(reset) for k, v in self._loss_meters.items()}
        return metrics

    def _postprocess_detections(self,
                                class_logits: torch.Tensor,
                                box_regression: torch.Tensor,
                                proposals: List[torch.Tensor],
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

    def _assign_targets_to_proposals(self,
                                     proposals: List[torch.Tensor],
                                     gt_boxes: List[torch.Tensor],
                                     gt_labels: List[torch.Tensor]):
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
            labels_in_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels


@Model.register("pretrained_detectron_faster_rcnn")
class PretrainedDetectronFasterRCNN(FasterRCNN):

    CATEGORIES = [
     'unlabeled',
     'person',
     'bicycle',
     'car',
     'motorcycle',
     'airplane',
     'bus',
     'train',
     'truck',
     'boat',
     'traffic light',
     'fire hydrant',
     'street sign',
     'stop sign',
     'parking meter',
     'bench',
     'bird',
     'cat',
     'dog',
     'horse',
     'sheep',
     'cow',
     'elephant',
     'bear',
     'zebra',
     'giraffe',
     'hat',
     'backpack',
     'umbrella',
     'shoe',
     'eye glasses',
     'handbag',
     'tie',
     'suitcase',
     'frisbee',
     'skis',
     'snowboard',
     'sports ball',
     'kite',
     'baseball bat',
     'baseball glove',
     'skateboard',
     'surfboard',
     'tennis racket',
     'bottle',
     'plate',
     'wine glass',
     'cup',
     'fork',
     'knife',
     'spoon',
     'bowl',
     'banana',
     'apple',
     'sandwich',
     'orange',
     'broccoli',
     'carrot',
     'hot dog',
     'pizza',
     'donut',
     'cake',
     'chair',
     'couch',
     'potted plant',
     'bed',
     'mirror',
     'dining table',
     'window',
     'desk',
     'toilet',
     'door',
     'tv',
     'laptop',
     'mouse',
     'remote',
     'keyboard',
     'cell phone',
     'microwave',
     'oven',
     'toaster',
     'sink',
     'refrigerator',
     'blender',
     'book',
     'clock',
     'vase',
     'scissors',
     'teddy bear',
     'hair drier',
     'toothbrush']

    def __init__(self,
                 rpn: Model,
                train_rpn: bool = False,
                pooler_sampling_ratio: int = 2,
                decoder_thresh: float = 0.1,
                decoder_nms_thresh: float = 0.5,
                decoder_detections_per_image: int = 100,
                matcher_high_thresh: float = 0.5,
                matcher_low_thresh: float = 0.5,
                allow_low_quality_matches: bool = True,
                batch_size_per_image: int = 64,
                balance_sampling_fraction: float = 0.25):
        feedforward = FeedForward(7 * 7 * 256, 2, [1024, 1024], nn.ReLU())
        encoder = FlattenEncoder(256, 7, 7, feedforward)
        vocab = Vocabulary({'labels': {k: 1 for k in PretrainedDetectronFasterRCNN.CATEGORIES}})
        super(PretrainedDetectronFasterRCNN, self)\
            .__init__(vocab, rpn, encoder,
                      pooler_resolution=7,
                      train_rpn=train_rpn,
                      pooler_sampling_ratio=pooler_sampling_ratio,
                      matcher_low_thresh=matcher_low_thresh,
                      matcher_high_thresh=matcher_high_thresh,
                      decoder_thresh=decoder_thresh,
                      decoder_nms_thresh=decoder_nms_thresh,
                      decoder_detections_per_image=decoder_detections_per_image,
                      allow_low_quality_matches=allow_low_quality_matches,
                      batch_size_per_image=batch_size_per_image,
                      balance_sampling_fraction=balance_sampling_fraction,
                      label_namespace='labels',
                      class_agnostic_bbox_reg=False)
        frcnn = fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
        self.cls_score.load_state_dict(frcnn.roi_heads.box_predictor.cls_score.state_dict())
        self.bbox_pred.load_state_dict(frcnn.roi_heads.box_predictor.bbox_pred.state_dict())

        feedforward._linear_layers[0].load_state_dict(frcnn.roi_heads.box_head.fc6.state_dict())
        feedforward._linear_layers[1].load_state_dict(frcnn.roi_heads.box_head.fc7.state_dict())
