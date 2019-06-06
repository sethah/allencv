local NUM_EPOCHS = 1;

local AUGMENTATION = [
            {
                "type": "keypoint_resize",
                "height": 512,
                "width": 512
            }, {
                "type": "normalize"
            }
        ];
local TRAIN_READER = {
        "type": "image_annotation",
        "augmentation": AUGMENTATION,
        "keypoint_name": "keypoints",
        "lazy": true
};

local BASE_ITERATOR = {
  "type": "basic",
  "batch_size": 1,
};

local RPN = {
    "type": "detectron_rpn",
    "anchor_sizes": [[32], [64], [128], [256], [1024]],
    "anchor_aspect_ratios": [[0.5, 1.0, 2.0], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0]],
    "requires_grad": false,
    "post_nms_top_n": 10000
};

{
  "dataset_reader": TRAIN_READER,
  "validation_dataset_reader": TRAIN_READER,
  "train_data_path": "allencv/tests/fixtures/data/image_annotation",
  "validation_data_path": "allencv/tests/fixtures/data/image_annotation",
  "model": {
    "type": "keypoint_rcnn",
    "rpn": RPN,
    "train_rpn": false,
    "roi_box_head": {
        "matcher_high_thresh": 0.0001,
        "matcher_low_thresh": 0.0,
        "decoder_thresh": 0.0,
        "decoder_detections_per_image": 10000,
        "batch_size_per_image": 10000000,
        "feature_extractor": {
            "type": "flatten",
            "input_channels": 256,
            "input_height": 7,
            "input_width": 7,
            "feedforward": {
                "input_dim": 7*7*256,
                "num_layers": 2,
                "hidden_dims": [256, 256],
                "activations": 'relu'
            }
        }
    },
    "roi_keypoint_head": {
        "feature_extractor": {
            "type": "feedforward",
            "input_channels": 256,
            "num_layers": 2,
            "hidden_channels": 64,
            "activations": "relu"
        },
        "pooler_resolution": 14
    },
    "num_keypoints": 4
  },
  "iterator": BASE_ITERATOR,
  "trainer": {
    "num_epochs": NUM_EPOCHS,
    "should_log_learning_rate": true,
    "cuda_device": -1,
    "optimizer": {
      "type": "adam",
      "lr": 1e-4,
    }
  }
}