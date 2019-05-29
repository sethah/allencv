local NUM_EPOCHS = 1;

local AUGMENTATION = [
            {
                "type": "resize",
                "height": 512,
                "width": 512
            }, {
                "type": "normalize"
            }
        ];
local TRAIN_READER = {
        "type": "image_annotation",
        "augmentation": AUGMENTATION,
        "lazy": true
};

local BASE_ITERATOR = {
  "type": "basic",
  "batch_size": 1,
};

local RPN = {
    "type": "detectron_rpn",
    "anchor_sizes": [[32], [64], [128], [256], [512]],
    "anchor_aspect_ratios": [[0.5, 1.0, 2.0], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0]],
};

{
  "dataset_reader": TRAIN_READER,
  "validation_dataset_reader": TRAIN_READER,
  "train_data_path": "allencv/tests/fixtures/data/image_annotation",
  "validation_data_path": "allencv/tests/fixtures/data/image_annotation",
  "model": {
    "type": "faster_rcnn",
    "rpn": RPN,
    "train_rpn": true,
    "matcher_high_thresh": 0.001,
    "matcher_low_thresh": 0.0,
    "batch_size_per_image": 10000000,
    "roi_feature_extractor": {
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
    },
    "num_labels": 4
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