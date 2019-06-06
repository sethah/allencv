import random

import torch
from allencv.modules.im2vec_encoders import FlattenEncoder
from allencv.modules.im2im_encoders import FeedforwardEncoder

from allencv.common.testing import AllenCvTestCase
from allencv.modules.object_detection.roi_heads import FasterRCNNROIHead, KeypointRCNNROIHead


class TestFasterRCNNROIHead(AllenCvTestCase):

    def test_forward(self):
        feature_channels = 64
        batch_size = 3
        num_labels = 9
        decoder_detections_per_image = 5
        features = [torch.randn(batch_size, feature_channels, i, i) for i in [64, 32, 16]]
        proposals_per_image = [random.randint(10, 20) for _ in range(batch_size)]
        n_proposals = sum(proposals_per_image)
        proposals = [torch.randint(200, (n, 4)).float() for n in proposals_per_image]
        image_sizes = [(512, 512) for _ in range(n_proposals)]
        pooler_resolution = 7
        feature_extractor = FlattenEncoder(feature_channels, pooler_resolution, pooler_resolution)
        roi_head = FasterRCNNROIHead(feature_extractor,
                                     decoder_nms_thresh=0.999,
                                     decoder_thresh=0.0,
                                     decoder_detections_per_image=decoder_detections_per_image)
        out = roi_head.forward(features, proposals, [(512, 512)])

        assert roi_head.get_output_dim() == (feature_channels * pooler_resolution ** 2)
        assert out.shape == (n_proposals, roi_head.get_output_dim())

        logits = torch.randn(n_proposals, num_labels)
        regression_preds = torch.randn(n_proposals, num_labels * 4)
        box_proposals, scores, class_preds = roi_head.postprocess_detections(logits,
                                                                             regression_preds,
                                                                             proposals, image_sizes)

        for proposal, score, pred, n in zip(box_proposals, scores, class_preds, proposals_per_image):
            # keypoint proposals are in (x, y, viz) format
            assert proposal.shape == (decoder_detections_per_image, 4)
            assert score.shape == (decoder_detections_per_image,)
            assert pred.shape == (decoder_detections_per_image,)


class TestKeypointRCNNROIHead(AllenCvTestCase):

    def test_forward(self):
        batch_size = 3
        pooler_resolution = 14
        feature_channels = 32
        num_keypoints = 4
        features = [torch.randn(batch_size, feature_channels, i, i) for i in [64, 32, 16]]
        proposals_per_image = [random.randint(1, 5) for _ in range(batch_size)]
        n_proposals = sum(proposals_per_image)
        proposals = [torch.randint(200, (n, 4)).float() for n in proposals_per_image]
        feature_extractor = FeedforwardEncoder(feature_channels,
                                               num_layers=2,
                                               hidden_channels=feature_channels,
                                               activations='relu')
        roi_head = KeypointRCNNROIHead(feature_extractor)
        image_sizes = [(512, 512) for _ in range(n_proposals)]
        features = roi_head.forward(features, proposals, image_sizes)
        assert features.shape == (n_proposals, feature_channels, pooler_resolution, pooler_resolution)

        logits = torch.randn(n_proposals, num_keypoints, pooler_resolution, pooler_resolution)
        keypoint_proposals, scores = roi_head.postprocess_detections(logits, proposals)
        assert len(keypoint_proposals) == batch_size
        assert len(scores) == batch_size
        for proposal, score, n in zip(keypoint_proposals, scores, proposals_per_image):
            # keypoint proposals are in (x, y, viz) format
            assert proposal.shape == (n, num_keypoints, 3)
            assert score.shape == (n, num_keypoints)
