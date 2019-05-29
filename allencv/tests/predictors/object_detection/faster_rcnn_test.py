import torch
import torch.nn as nn

from allencv.models.object_detection import FasterRCNN
from allencv.modules.im2vec_encoders import FlattenEncoder

from allencv.common.testing import AllenCvTestCase
from allencv.data.dataset_readers import ImageAnnotationReader
from allencv.predictors.object_detection import FasterRCNNPredictor
from allencv.models.object_detection import RPN
from allencv.modules.image_encoders import ResnetEncoder, FPN

from allennlp.modules import FeedForward


class TestFasterRCNN(AllenCvTestCase):

    def test_predictor(self):
        batch_size = 3
        im = torch.randn(batch_size, 3, 224, 224)
        im_sizes = torch.tensor([im.shape[-2:] for _ in range(im.shape[0])]).view(3, -1)
        backbone = ResnetEncoder('resnet18')
        fpn_out_channels = 256
        fpn_backbone = FPN(backbone, fpn_out_channels)
        anchor_sizes = [[32], [64], [128], [256], [512]]
        anchor_strides = [[0.5, 1.0, 2.0], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0],
                          [0.5, 1.0, 2.0], [0.5, 1.0, 2.0]]
        fpn_pos_nms_top_n = 400
        rpn = RPN(fpn_backbone,
                  anchor_sizes=anchor_sizes,
                  anchor_strides=anchor_strides,
                  fpn_post_nms_top_n=fpn_pos_nms_top_n,
                  match_thresh_high=0.001,
                  match_thresh_low=0.0,
                  batch_size_per_image=10000000)
        pooler_resolution = 7
        roi_num_features = pooler_resolution * pooler_resolution * fpn_out_channels
        feedforward = FeedForward(roi_num_features, num_layers=2, hidden_dims=[32, 32],
                                  activations=nn.ReLU())
        encoder = FlattenEncoder(fpn_out_channels, pooler_resolution, pooler_resolution,
                                 feedforward)
        frcnn = FasterRCNN(None, rpn, encoder,
                           num_labels=5,
                           decoder_thresh=0.0,
                           decoder_nms_thresh=0.01,
                           batch_size_per_image=1000000)
        reader = ImageAnnotationReader()
        predictor = FasterRCNNPredictor(frcnn, reader)
        predicted = predictor.predict(AllenCvTestCase.FIXTURES_ROOT / "data" / "image_annotation" / "images" / "00001.jpg")
        assert len(predicted['scores']) == len(predicted['labels'])
