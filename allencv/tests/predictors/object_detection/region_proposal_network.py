from allencv.common.testing import AllenCvTestCase
from allencv.data.dataset_readers import ImageAnnotationReader
from allencv.predictors import ImagePredictor
from allencv.models.object_detection import RPN
from allencv.modules.image_encoders import ResnetEncoder, FPN


class TestRegionProposalNetwork(AllenCvTestCase):

    def test_predictor(self):
        backbone = ResnetEncoder('resnet18')
        fpn_out_channels = 256
        fpn_backbone = FPN(backbone, fpn_out_channels)
        anchor_sizes = [[32], [64], [128], [256], [512]]
        anchor_aspect_ratios = [[0.5, 1.0, 2.0], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0],
                          [0.5, 1.0, 2.0], [0.5, 1.0, 2.0]]
        fpn_post_nms_top_n = 400
        rpn = RPN(fpn_backbone,
                  anchor_sizes=anchor_sizes,
                  anchor_aspect_ratios=anchor_aspect_ratios,
                  fpn_post_nms_top_n=fpn_post_nms_top_n)
        reader = ImageAnnotationReader()
        predictor = ImagePredictor(rpn, reader)
        predicted = predictor.predict(AllenCvTestCase.FIXTURES_ROOT / "data" / "image_annotation" / "images" / "00001.jpg")
        assert all([len(x) == 4 for x in predicted['proposals']])

