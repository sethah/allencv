import torch
import torch.nn as nn

from torchvision.models.detection.keypoint_rcnn import keypointrcnn_resnet50_fpn

from allencv.common.testing import ModelTestCase
from allencv.models.object_detection import RPN, RCNN, PretrainedDetectronFasterRCNN
from allencv.models.object_detection.roi_heads import FasterRCNNROIHead, KeypointRCNNROIHead
from allencv.modules.image_encoders import ResnetEncoder, FPN
from allencv.modules.im2vec_encoders import FlattenEncoder

from allennlp.data import Vocabulary

from allencv.common.testing import AllenCvTestCase, ModelTestCase
from allencv.modules.im2im_encoders import FeedforwardEncoder
from allencv.data.dataset_readers import ImageAnnotationReader
from allencv.models.object_detection import RPN
from allencv.modules.image_encoders import ResnetEncoder, FPN

from allennlp.modules import FeedForward


class TestFasterRCNN(ModelTestCase):

    def test_forward(self):
        reader = ImageAnnotationReader()
        instances = reader.read(self.FIXTURES_ROOT / "data" / "image_annotation")
        batch_size = 1
        im = torch.randn(batch_size, 3, 224, 224)
        im_sizes = torch.tensor([im.shape[-2:] for _ in range(im.shape[0])]).view(batch_size, -1)
        backbone = ResnetEncoder('resnet18')
        fpn_out_channels = 256
        fpn_backbone = FPN(backbone, fpn_out_channels)
        anchor_sizes = [[32], [64], [128], [256], [512]]
        anchor_aspect_ratios = [[0.5, 1.0, 2.0], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0],
                          [0.5, 1.0, 2.0], [0.5, 1.0, 2.0]]
        fpn_pos_nms_top_n = 400
        rpn = RPN(fpn_backbone,
                  anchor_sizes=anchor_sizes,
                  anchor_aspect_ratios=anchor_aspect_ratios,
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
        tmodel = keypointrcnn_resnet50_fpn()
        vocab = Vocabulary.from_instances(instances)
        box_head = FasterRCNNROIHead(vocab, encoder)
        kp_feedforward = FeedForward(roi_num_features, num_layers=2, hidden_dims=[32, 32],
                                  activations=nn.ReLU())
        kp_encoder = FeedforwardEncoder(256, 5, [512, 512], 'relu')
        kp_head = KeypointRCNNROIHead(kp_encoder, 17)
        for n, p in tmodel.named_parameters():
            print(n)
        print('asdf')
        kp_rcnn = RCNN(None, rpn, box_head, roi_keypoint_head=kp_head)
        for n, p in kp_rcnn.named_parameters():
            print(n)
        # frcnn = RCNN(None, rpn, box_head, kp_head)
        # boxes = torch.tensor([0, 0, 20, 13, 10, 10, 40, 21]).view(1, 2, 4).float()
        # box_classes = torch.tensor([1, 1.]).view(1, 2, 1)
        # keypoint_positions = torch.randint(200, (1, 17, 3)).float()
        # keypoint_positions[:, :, -1] = 2
        # out = frcnn.forward(im, im_sizes, boxes, box_classes, keypoint_positions)
        # print(out.keys())
        # print(out['keypoint_logits'].shape)
        # print(out['keypoint_loss'])
        inst = instances[0]
        inst.index_fields(vocab)
        # print(inst.as_tensor_dict())
        out = kp_rcnn.forward_on_instance(inst)

