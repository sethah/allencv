import logging
from typing import List

import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.nn import InitializerApplicator
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from allencv.models.object_detection import RCNN
from allencv.modules.im2vec_encoders import FlattenEncoder
from allencv.modules.object_detection.roi_heads import FasterRCNNROIHead

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

TensorList = List[torch.Tensor]


@Model.register("faster_rcnn")
class FasterRCNN(RCNN):

    def __init__(self,
                 vocab: Vocabulary,
                 rpn: Model,
                 roi_box_head: FasterRCNNROIHead,
                 train_rpn: bool = False,
                 num_labels: int = None,
                 label_namespace: str = "labels",
                 class_agnostic_bbox_reg: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super(FasterRCNN, self).__init__(vocab, rpn, roi_box_head,
                                         train_rpn=train_rpn,
                                         num_keypoints=None,
                                         roi_keypoint_head=None,
                                         num_labels=num_labels,
                                         label_namespace=label_namespace,
                                         class_agnostic_bbox_reg=class_agnostic_bbox_reg,
                                         initializer=initializer)


@Model.register("pretrained_detectron_faster_rcnn")
class PretrainedDetectronFasterRCNN(RCNN):

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
        box_roi_head = FasterRCNNROIHead(encoder,
                                         pooler_resolution=7,
                                         pooler_sampling_ratio=pooler_sampling_ratio,
                                         matcher_low_thresh=matcher_low_thresh,
                                         matcher_high_thresh=matcher_high_thresh,
                                         decoder_thresh=decoder_thresh,
                                         decoder_nms_thresh=decoder_nms_thresh,
                                         decoder_detections_per_image=decoder_detections_per_image,
                                         allow_low_quality_matches=allow_low_quality_matches,
                                         batch_size_per_image=batch_size_per_image,
                                         balance_sampling_fraction=balance_sampling_fraction)
        super(PretrainedDetectronFasterRCNN, self).__init__(vocab,
                                                            rpn,
                                                            box_roi_head,
                                                            train_rpn=train_rpn)
        frcnn = fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
        self._box_classifier.load_state_dict(frcnn.roi_heads.box_predictor.cls_score.state_dict())
        self._bbox_pred.load_state_dict(frcnn.roi_heads.box_predictor.bbox_pred.state_dict())

        # pylint: disable = protected-access
        feedforward._linear_layers[0].load_state_dict(frcnn.roi_heads.box_head.fc6.state_dict())
        feedforward._linear_layers[1].load_state_dict(frcnn.roi_heads.box_head.fc7.state_dict())

        for p in rpn.parameters():
            p.requires_grad = train_rpn
