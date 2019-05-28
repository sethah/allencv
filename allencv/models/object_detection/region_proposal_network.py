import logging
from overrides import overrides
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet

from allennlp.common import Params
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import Average

from allencv.modules.image_encoders import ImageEncoder, ResnetEncoder, FPN
from allencv.models.object_detection import utils as object_utils
from allencv.modules.image_encoders.resnet_encoder import Bottleneck

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import ImageList
from maskrcnn_benchmark.modeling.rpn.loss import RPNLossComputation, generate_rpn_labels
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from maskrcnn_benchmark.modeling.rpn.anchor_generator import AnchorGenerator
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.rpn.inference import RPNPostProcessor
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

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
    anchor_strides: ``List[int]``
        How far apart each anchor box center is from its neighbors.
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
    straddle_thresh: ``int``
        How many pixels of an anchor box can be outside the image bounds before it is no
        longer considered valid.
    """

    def __init__(self,
                 backbone: ImageEncoder,
                 positive_fraction: float = 0.5,
                 match_thresh_low: float = 0.3,
                 match_thresh_high: float = 0.7,
                 anchor_sizes: List[int] = (128, 256, 512),
                 anchor_aspect_ratios: List[float] = (0.5, 1.0, 2.0),
                 anchor_strides: List[int] = (8, 16, 32),
                 batch_size_per_image: int = 256,
                 pre_nms_top_n: int = 6000,
                 post_nms_top_n: int = 300,
                 nms_thresh: int = 0.7,
                 min_size: int = 0,
                 fpn_post_nms_top_n: int = 1000,
                 fpn_post_nms_per_batch: int = True,
                 allow_low_quality_matches: bool = True,
                 straddle_thresh: int = 0,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(RPN, self).__init__(None)

        # the BoxCoder just converts the relative regression offsets into absolute
        # coordinates
        box_coder = BoxCoder(weights=(1., 1., 1., 1.))

        # sampler is responsible for selecting a subset of anchor boxes for computing the loss
        # this makes sure each batch has reasonable balance of foreground/background labels
        # it selects `batch_size_per_image` total boxes
        sampler = BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)

        # matcher decides if an anchor box is a foreground or background based on how much
        # it overlaps with the nearest target box
        matcher = Matcher(match_thresh_high, match_thresh_low,
                          allow_low_quality_matches=allow_low_quality_matches)

        # the decoder will choose the highest scoring foreground anchor boxes and then perform
        # non-max suppression on those to eliminate duplicates. After that it will choose
        # a further subset of the ones that survived nms as proposals from the RPN
        self.decoder = RPNPostProcessor(pre_nms_top_n=pre_nms_top_n,
                                        post_nms_top_n=post_nms_top_n,
                                        nms_thresh=nms_thresh,
                                        min_size=min_size,
                                        box_coder=box_coder,
                                        fpn_post_nms_top_n=fpn_post_nms_top_n,
                                        fpn_post_nms_per_batch=fpn_post_nms_per_batch)
        self.backbone = backbone
        self.loss_evaluator = RPNLossComputation(matcher,
                                                 sampler,
                                                 box_coder,
                                                 generate_rpn_labels)
        self.anchor_generator = AnchorGenerator(anchor_sizes, anchor_aspect_ratios, anchor_strides,
                                                straddle_thresh=straddle_thresh)
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
        anchors_bbox = []
        for anchor_bbox in anchors:
            anchors_bbox.append([b.bbox for b in anchor_bbox])

        out = {'features': features, 'objectness': objectness,
               'rpn_box_regression': rpn_box_regression, 'anchors_bbox': anchors_bbox,
               'anchors_sizes': im_sizes}
        if boxes is not None:
            box_list: List[BoxList] = object_utils.padded_tensor_to_box_list(boxes, image_sizes)
            loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
                anchors, objectness, rpn_box_regression, box_list
            )
            out["loss_objectness"] = loss_objectness
            out["loss_rpn_box_reg"] = loss_rpn_box_reg
            self._loss_meters['rpn_cls_loss'](loss_objectness.item())
            self._loss_meters['rpn_reg_loss'](loss_rpn_box_reg.item())
            out["loss"] = loss_objectness + 10*loss_rpn_box_reg
        return out

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # you get a list of proposal boxes for each image in the batch
        anchors = []
        for i, anchor_bbox in enumerate(output_dict['anchors_bbox']):
            anchors.append([BoxList(bbox, output_dict['anchors_sizes'][i]) for bbox in anchor_bbox])
        proposals: List[BoxList] = self.decoder(anchors, output_dict['objectness'],
                                                output_dict['rpn_box_regression'])
        proposal_scores = [b.get_field("objectness").unsqueeze(1) for b in proposals]
        proposals: torch.Tensor = object_utils.pad_tensors([p.bbox for p in proposals])
        proposal_scores: torch.Tensor = object_utils.pad_tensors(proposal_scores)[:, :, 0]
        output_dict['proposals'] = proposals
        output_dict['proposal_scores'] = proposal_scores
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {k: v.get_metric(reset) for k, v in self._loss_meters.items()}
        return metrics


@Model.register("pretrained")
class PretrainedRPN(RPN):

    @classmethod
    def from_params(cls, params: Params):
        model = load_archive(params.pop("archive_file")).model
        requires_grad = params.pop("requires_grad")
        for p in model.parameters():
            p.requires_grad = requires_grad
        return model


@Model.register("detectron_rpn")
class PretrainedDetectronRPN(RPN):

    def __init__(self,
                 anchor_sizes: List[int] = (128, 256, 512),
                 anchor_aspect_ratios: List[float] = (0.5, 1.0, 2.0),
                 anchor_strides: List[int] = (8, 16, 32),
                 batch_size_per_image: int = 256):
        # backbone = ResnetEncoder('resnet50')
        # this is Resnet50 but with the stride matching the detectron pretrained model
        resnet = ResNet(Bottleneck, [3, 4, 6, 3])
        backbone = ResnetEncoder(resnet)
        fpn = FPN(backbone, 256)
        super(PretrainedDetectronRPN, self).__init__(fpn, anchor_strides=anchor_strides,
                                                     anchor_aspect_ratios=anchor_aspect_ratios,
                                                     anchor_sizes=anchor_sizes,
                                                     batch_size_per_image=batch_size_per_image)
        # TODO: don't rely on their silly config?
        cfg.MODEL.WEIGHT = "catalog://Caffe2Detectron/COCO/35857345/e2e_faster_rcnn_R-50-FPN_1x"
        checkpointer = DetectronCheckpointer(cfg, None, save_dir=None)
        f = checkpointer._load_file(cfg.MODEL.WEIGHT)
        rpn_dict = {k.replace("rpn.head.", ""): v for k, v in f['model'].items() if
                    k.startswith('rpn.head')}
        self.load_state_dict(rpn_dict, strict=False)
        backbone_dict = {k: v for k, v in f['model'].items() if k.startswith('layer')}
        backbone_dict['stem.conv1.bias'] = f['model']['conv1.bias']
        backbone_dict['stem.conv1.weight'] = f['model']['conv1.weight']
        backbone_dict['stem.bn1.bias'] = f['model']['bn1.bias']
        backbone_dict['stem.bn1.weight'] = f['model']['bn1.weight']
        resnet_dict = {k: v for k, v in backbone_dict.items()}
        backbone.load_state_dict(resnet_dict, strict=False)
        self._load_fpn_detectron_state(fpn, f['model'])

        def deactivate_batchnorm(m):
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.
                m.eps = 0.
                m.running_mean.fill_(0.0)
                m.running_var.fill_(1.0)
                # TODO: this will be undone with any model.training() call
                m.eval()
        self.apply(deactivate_batchnorm)

    def _load_fpn_detectron_state(self, fpn, state: Dict[str, torch.Tensor]):
        for i in range(4):
            prefix = f'fpn_inner{i + 1}'
            state_dict = {'weight': state[f'{prefix}.weight'],
                          'bias': state[f'{prefix}.bias']}
            fpn._convert_layers[i].load_state_dict(state_dict)
        for i in range(4):
            prefix = f'fpn_layer{i + 1}'
            state_dict = {'weight': state[f'{prefix}.weight'],
                          'bias': state[f'{prefix}.bias']}
            fpn._combine_layers[-(i + 1)].load_state_dict(state_dict)


