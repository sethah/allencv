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

{
  "dataset_reader": TRAIN_READER,
  "validation_dataset_reader": TRAIN_READER,
  "train_data_path": "allencv/tests/fixtures/data/image_annotation",
  "validation_data_path": "allencv/tests/fixtures/data/image_annotation",
  "model": {
    "type": "rpn",
    "backbone": {
        "type": "feature_pyramid",
        "backbone": {
            "type": "resnet_encoder",
            "model_str": "resnet34",
            "pretrained": "false",
            "requires_grad": "true"
        },
        "output_channels": 256
    },
    "anchor_sizes": [64, 128, 256, 512],
    "anchor_strides": [4, 8, 16, 32],
    "match_thresh_high": 0.001,
    "match_thresh_low": 0.0,
    "batch_size_per_image": 10000000,
    straddle_thresh: 2000
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