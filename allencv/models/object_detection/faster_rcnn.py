import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator

from allencv.models.region_proposal_network import RPN
from allencv.modules.im2vec_encoders import Im2VecEncoder
from allennlp.training.metrics import Average

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.roi_heads.box_head.loss import FastRCNNLossComputation
from maskrcnn_benchmark.modeling.roi_heads.box_head.inference import PostProcessor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("faster_rcnn")
class FasterRCNN(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 rpn: RPN,
                 roi_feature_extractor: Im2VecEncoder,
                 num_labels: int = None,
                 label_namespace: str = "labels",
                 pooler_resolution: int = 7,
                 pooler_scales: Tuple[float] = (0.25, 0.125, 0.0625, 0.03125),
                 pooler_sampling_ratio: int = 2,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(FasterRCNN, self).__init__(vocab)
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
        class_agnostic_bbox_reg = True
        num_bbox_reg_classes = 2 if class_agnostic_bbox_reg else self._num_labels
        representation_size = roi_feature_extractor.get_output_dim()
        self.cls_score = nn.Linear(representation_size, self._num_labels)
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)
        self.decoder = PostProcessor()
        matcher = Matcher(0.5, 0.5, allow_low_quality_matches=True)
        batch_size_per_image = 512
        sampler = BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction=.25)
        box_coder = BoxCoder(weights=(1., 1., 1., 1.))
        self.roi_loss_evaluator = FastRCNNLossComputation(
            matcher,
            sampler,
            box_coder,
            class_agnostic_bbox_reg
        )
        self._loss_meters = {'roi_cls_loss': Average(), 'roi_reg_loss': Average(),
                             'rpn_cls_loss': Average(), 'rpn_reg_loss': Average()}
        self.decoder = PostProcessor(score_thresh=0.1, nms=0.5, detections_per_img=100,
                           cls_agnostic_bbox_reg=True)
        initializer(self)

    def forward(self,
                image: torch.Tensor,  # (batch_size, c, h, w)
                image_sizes: torch.Tensor,  # (batch_size, 2)
                boxes: torch.Tensor = None,  # (batch_size, s, 4)
                box_classes: torch.Tensor = None):
        rpn_out = self.rpn.forward(image, image_sizes, boxes)
        features: List[torch.Tensor] = rpn_out['features']
        proposals: torch.Tensor = rpn_out['proposals']
        proposals_: List[BoxList] = self.rpn._padded_tensor_to_box_list(proposals, image_sizes)
        if boxes is not None:
            # subsample each set of proposals to a pre-specified number, e.g. 512
            boxlist = self.rpn._padded_tensor_to_box_list(boxes, image_sizes, labels=box_classes)
            with torch.no_grad():
                sampled_proposals_ = self.roi_loss_evaluator.subsample(proposals_, boxlist)

        # roi pool and pass through feature extractor
        # (b, num_proposal_regions_in_batch, 14, 14)
        pooled = self.pooler(features, sampled_proposals_)

        # pooler, combines logits from all images in the batch to one large batch
        # you would need to split the tensor by the number of proposals in each image to
        # recover the per-image logits
        region_features = self.feature_extractor(pooled)
        class_logits = self.cls_score(region_features)
        #[(13, 4), (17, 4)] -> (2, 17, 4)
        split_logits = class_logits.split([len(p) for p in sampled_proposals_])
        split_logits = self.rpn._pad_tensors(split_logits)
        box_regression = self.bbox_pred(region_features)
        split_box_regression = box_regression.split([len(p) for p in sampled_proposals_])
        split_box_regression = self.rpn._pad_tensors(split_box_regression)
        out = {'class_logits': split_logits,
               'box_regression': split_box_regression,
               'proposals': proposals}
        # decoder splits the input tensors by the sizes stored in `proposals_` to get
        # the per-image predictions

        if not self.training:
            decoded: List[BoxList] = self.decoder.forward((class_logits, box_regression), sampled_proposals_)
            # scores = decoded[0].get_field("scores")
            out['scores'] = self.rpn._pad_tensors([d.get_field("scores") for d in decoded])
            out['labels'] = self.rpn._pad_tensors([d.get_field("labels") for d in decoded])
            decoded = self.rpn._pad_tensors([d.bbox for d in decoded])
            out['decoded'] = decoded

        if boxes is not None:
            rpn_classifier_loss = rpn_out['loss_objectness']
            rpn_regression_loss = rpn_out['loss_rpn_box_reg']
            classifier_loss, regression_loss = self.roi_loss_evaluator([class_logits], [box_regression])
            out['loss'] = 0.003*classifier_loss + regression_loss + 0.1*rpn_classifier_loss + rpn_regression_loss
            self._loss_meters['rpn_cls_loss'](rpn_out['loss_objectness'].item())
            self._loss_meters['rpn_reg_loss'](rpn_out['loss_rpn_box_reg'].item())
            self._loss_meters['roi_cls_loss'](classifier_loss.item())
            self._loss_meters['roi_reg_loss'](regression_loss.item())
        return out

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {k: v.get_metric(reset) for k, v in self._loss_meters.items()}
        return metrics
