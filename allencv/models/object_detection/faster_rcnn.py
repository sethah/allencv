import logging
from overrides import overrides
import numpy as np
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import Average

from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import \
    BalancedPositiveNegativeSampler
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.roi_heads.box_head.inference import PostProcessor
from maskrcnn_benchmark.modeling.roi_heads.box_head.loss import FastRCNNLossComputation
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

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
    pooler_scales: ``List[float]``
        Scales for each pooler.
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
                 pooler_scales: Tuple[float] = (0.25, 0.125, 0.0625, 0.03125),
                 pooler_sampling_ratio: int = 2,
                 decoder_thresh: float = 0.1,
                 decoder_nms_thresh: float = 0.5,
                 decoder_detections_per_image: int = 100,
                 matcher_high_thresh: float = 0.5,
                 matcher_low_thresh: float = 0.5,
                 allow_low_quality_matches: bool = True,
                 batch_size_per_image: int = 64,
                 balance_sampling_fraction: float = 0.25,
                 class_agnostic_bbox_reg: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(FasterRCNN, self).__init__(vocab)
        self._train_rpn = train_rpn
        self.rpn = rpn
        self.pooler = Pooler(
            output_size=(pooler_resolution, pooler_resolution),
            scales=pooler_scales,
            sampling_ratio=pooler_sampling_ratio,
        )
        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=label_namespace)
        self.feature_extractor = roi_feature_extractor
        num_bbox_reg_classes = 2 if class_agnostic_bbox_reg else self._num_labels
        representation_size = roi_feature_extractor.get_output_dim()
        self.cls_score = nn.Linear(representation_size, self._num_labels)
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)
        matcher = Matcher(high_threshold=matcher_high_thresh,
                          low_threshold=matcher_low_thresh,
                          allow_low_quality_matches=allow_low_quality_matches)
        sampler = BalancedPositiveNegativeSampler(batch_size_per_image,
                                                  positive_fraction=balance_sampling_fraction)
        box_coder = BoxCoder(weights=(1., 1., 1., 1.))
        self.roi_loss_evaluator = FastRCNNLossComputation(
            matcher,
            sampler,
            box_coder,
            class_agnostic_bbox_reg
        )
        self._loss_meters = {'roi_cls_loss': Average(), 'roi_reg_loss': Average(),
                             'rpn_cls_loss': Average(), 'rpn_reg_loss': Average()}
        self.decoder = PostProcessor(score_thresh=decoder_thresh,
                                     nms=decoder_nms_thresh,
                                     detections_per_img=decoder_detections_per_image,
                                     cls_agnostic_bbox_reg=class_agnostic_bbox_reg)
        initializer(self)

    def forward(self,
                image: torch.Tensor,  # (batch_size, c, h, w)
                image_sizes: torch.Tensor,  # (batch_size, 2)
                boxes: torch.Tensor = None,  # (batch_size, s, 4)
                box_classes: torch.Tensor = None):
        rpn_out = self.rpn.forward(image, image_sizes, boxes)
        rpn_out = self.rpn.decode(rpn_out)
        features: List[torch.Tensor] = rpn_out['features']
        proposals: List[BoxList] = object_utils.padded_tensor_to_box_list(rpn_out['proposals'],
                                                                       image_sizes)
        if boxes is not None:
            # subsample each set of proposals to a pre-specified number, e.g. 512
            boxlist = object_utils.padded_tensor_to_box_list(boxes, image_sizes, labels=box_classes)
            with torch.no_grad():
                sampled_proposals = self.roi_loss_evaluator.subsample(proposals, boxlist)
        else:
            sampled_proposals = proposals

        # roi pool and pass through feature extractor
        # (b, num_proposal_regions_in_batch, 14, 14)
        pooled = self.pooler(features, sampled_proposals)

        # pooler, combines logits from all images in the batch to one large batch
        # you would need to split the tensor by the number of proposals in each image to
        # recover the per-image logits
        region_features = self.feature_extractor(pooled)
        class_logits = self.cls_score(region_features)
        box_regression = self.bbox_pred(region_features)
        out = {'class_logits': class_logits,
               'box_regression': box_regression,
               'proposals': sampled_proposals}

        if boxes is not None:
            rpn_classifier_loss = rpn_out['loss_objectness']
            rpn_regression_loss = rpn_out['loss_rpn_box_reg']
            classifier_loss, regression_loss = self.roi_loss_evaluator([class_logits], [box_regression])
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
        decoded: List[BoxList] = self.decoder.forward((output_dict['class_logits'],
                                                       output_dict['box_regression']),
                                                     output_dict['proposals'])
        output_dict['scores'] = object_utils.pad_tensors([d.get_field("scores") for d in decoded])
        output_dict['labels'] = object_utils.pad_tensors([d.get_field("labels") for d in decoded])
        output_dict['decoded'] = object_utils.pad_tensors([d.bbox for d in decoded])
        output_dict.pop("proposals")
        all_predictions = output_dict['labels']
        all_predictions = all_predictions.cpu().data.numpy()
        if all_predictions.ndim == 2:
            predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        if self.vocab is not None:
            for predictions in predictions_list:
                tags = [self.vocab.get_token_from_index(x, namespace="labels")
                        for x in predictions]
                all_tags.append(tags)
            output_dict['class'] = all_tags
        # split_logits = class_logits.split([len(p) for p in sampled_proposals])
        # # [(13, 4), (17, 4)] -> (2, 17, 4)
        # split_logits = self.rpn._pad_tensors(split_logits)
        # split_box_regression = box_regression.split([len(p) for p in sampled_proposals])
        # split_box_regression = self.rpn._pad_tensors(split_box_regression)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {k: v.get_metric(reset) for k, v in self._loss_meters.items()}
        return metrics

COCO_CATEGORIES = [
        "__background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]


@Model.register("pretrained_detectron_faster_rcnn")
class PretrainedDetectronFasterRCNN(FasterRCNN):

    def __init__(self,
                 rpn: Model,
                train_rpn: bool = False,
                pooler_scales: Tuple[float] = (0.25, 0.125, 0.0625, 0.03125),
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
        vocab = Vocabulary({'labels': {k: 1 for k in COCO_CATEGORIES}})
        super(PretrainedDetectronFasterRCNN, self).__init__(vocab, rpn, encoder,
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
                                                            pooler_scales=pooler_scales,
                                                            balance_sampling_fraction=balance_sampling_fraction,
                                                            label_namespace='labels',
                                                    class_agnostic_bbox_reg=False)
        # TODO: don't rely on their silly config?
        cfg.MODEL.WEIGHT = "catalog://Caffe2Detectron/COCO/35857345/e2e_faster_rcnn_R-50-FPN_1x"
        checkpointer = DetectronCheckpointer(cfg, None, save_dir=None)
        f = checkpointer._load_file(cfg.MODEL.WEIGHT)
        feedforward._linear_layers.load_state_dict({
            '0.weight': f['model']['fc6.weight'],
            '0.bias': f['model']['fc6.bias'],
            '1.weight': f['model']['fc7.weight'],
            '1.bias': f['model']['fc7.bias']
        })
        self.cls_score.load_state_dict({'weight': f['model']['cls_score.weight'],
                                        'bias': f['model']['cls_score.bias']})
        self.bbox_pred.load_state_dict({'weight': f['model']['bbox_pred.weight'],
                                        'bias': f['model']['bbox_pred.bias']})

