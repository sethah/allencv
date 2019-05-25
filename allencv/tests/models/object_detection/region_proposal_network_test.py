import torch

from allencv.common.testing import AllenCvTestCase

from allencv.modules.image_encoders import ResnetEncoder, FPN
from allencv.models.object_detection import PretrainedDetectronRPN, RPN


class TestRegionProposalNetwork(AllenCvTestCase):

    def test_forward(self):
        im = torch.randn(3, 3, 224, 224)
        im_sizes = torch.tensor([im.shape[-2:] for i in range(im.shape[0])]).view(3, -1)
        backbone = ResnetEncoder('resnet50')
        fpn_backbone = FPN(backbone, 256)
        anchor_sizes = [32, 64, 128, 256, 512]
        anchor_strides = [4, 8, 16, 32, 64]
        rpn = RPN(fpn_backbone, anchor_sizes=anchor_sizes, anchor_strides=anchor_strides)
        out = rpn.forward(im, im_sizes)
        assert len(out['features']) == len(fpn_backbone.get_output_channels())
        assert len(out['objectness']) == len(fpn_backbone.get_output_channels())
        assert len(out['rpn_box_regression']) == len(fpn_backbone.get_output_channels())
        assert len(out['anchors']) == im.shape[0]
        assert len(out['anchors'][0]) == len(anchor_sizes)

    def test_decode(self):
        im = torch.randn(3, 3, 224, 224)
        im_sizes = torch.tensor([im.shape[-2:] for i in range(im.shape[0])]).view(3, -1)
        backbone = ResnetEncoder('resnet50')
        fpn_backbone = FPN(backbone, 256)
        anchor_sizes = [32, 64, 128, 256, 512]
        anchor_strides = [4, 8, 16, 32, 64]
        rpn = RPN(fpn_backbone,
                  anchor_sizes=anchor_sizes,
                  anchor_strides=anchor_strides,
                  fpn_post_nms_top_n=50)
        out = rpn.forward(im, im_sizes)
        decoded = rpn.decode(out)
        assert decoded['proposals'].shape[0] == 3
        assert decoded['proposals'].shape[2] == 4
        assert decoded['proposal_scores'].shape[0] == 3
        assert decoded['proposal_scores'].shape[1] <= 50
        assert decoded['proposal_scores'].shape[1] == decoded['proposals'].shape[1]

    def test_detectron_rpn(self):
        im = torch.randn(3, 3, 224, 224)
        im_sizes = torch.tensor([im.shape[-2:] for i in range(im.shape[0])]).view(3, -1)
        anchor_sizes = [32, 64, 128, 256, 512]
        anchor_strides = [4, 8, 16, 32, 64]
        rpn = PretrainedDetectronRPN(anchor_sizes=anchor_sizes, anchor_strides=anchor_strides)
        out = rpn.forward(im, im_sizes)
        assert len(out['features']) == len(rpn.backbone.get_output_channels())
        assert len(out['objectness']) == len(rpn.backbone.get_output_channels())
        assert len(out['rpn_box_regression']) == len(rpn.backbone.get_output_channels())
        assert len(out['anchors']) == im.shape[0]
        assert len(out['anchors'][0]) == len(anchor_sizes)


