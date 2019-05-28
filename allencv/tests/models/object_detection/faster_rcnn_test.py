import torch
import torch.nn as nn

from allencv.common.testing import ModelTestCase
from allencv.models.object_detection import RPN, FasterRCNN, PretrainedDetectronFasterRCNN, PretrainedDetectronRPN
from allencv.modules.image_encoders import ResnetEncoder, FPN
from allencv.modules.im2vec_encoders import FlattenEncoder

from allennlp.modules import FeedForward


class TestFasterRCNN(ModelTestCase):

    def test_forward(self):
        batch_size = 3
        im = torch.randn(batch_size, 3, 224, 224)
        im_sizes = torch.tensor([im.shape[-2:] for _ in range(im.shape[0])]).view(3, -1)
        backbone = ResnetEncoder('resnet50')
        fpn_out_channels = 256
        fpn_backbone = FPN(backbone, fpn_out_channels)
        anchor_sizes = [32, 64, 128, 256, 512]
        anchor_strides = [4, 8, 16, 32, 64]
        fpn_pos_nms_top_n = 400
        rpn = RPN(fpn_backbone, anchor_sizes=anchor_sizes, anchor_strides=anchor_strides,
                  fpn_post_nms_top_n=fpn_pos_nms_top_n)
        pooler_resolution = 7
        roi_num_features = pooler_resolution * pooler_resolution * fpn_out_channels
        feedforward = FeedForward(roi_num_features, num_layers=2, hidden_dims=[1024, 1024],
                                  activations=nn.ReLU())
        encoder = FlattenEncoder(fpn_out_channels, pooler_resolution, pooler_resolution,
                                 feedforward)
        num_labels = 4
        decoder_detections_per_image = 73
        frcnn = FasterRCNN(None, rpn, encoder, num_labels=num_labels,
                           decoder_detections_per_image=decoder_detections_per_image)
        out = frcnn.forward(im, im_sizes)
        assert list(out['class_logits'].shape) == [fpn_pos_nms_top_n, num_labels]

        decoded = frcnn.decode(out)
        assert list(decoded['decoded'].shape) == [batch_size, decoder_detections_per_image, 4]

    def test_pretrained(self):
        batch_size = 3
        im = torch.randn(batch_size, 3, 224, 224)
        im_sizes = torch.tensor([im.shape[-2:] for _ in range(im.shape[0])]).view(3, -1)
        anchor_sizes = [32, 64, 128, 256, 512]
        anchor_strides = [4, 8, 16, 32, 64]
        decoder_detections_per_image = 73
        rpn = PretrainedDetectronRPN(anchor_sizes=anchor_sizes, anchor_strides=anchor_strides)
        frcnn = PretrainedDetectronFasterRCNN(rpn,
                                              decoder_thresh=0.0,
                                              decoder_detections_per_image=decoder_detections_per_image)
        out = frcnn.forward(im, im_sizes)
        assert out['class_logits'].shape[1] == 81
        decoded = frcnn.decode(out)
        assert list(decoded['decoded'].shape) == [batch_size, decoder_detections_per_image, 4]


