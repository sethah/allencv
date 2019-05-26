local NUM_EPOCHS = 5;

local image_height = 224;
local image_width = 224;
local AUGMENTATION = [
            {
                "type": "resize",
                "height": image_height,
                "width": image_width
            }, {
                "type": "normalize"
            }
        ];
local READER = {
        "type": "paired_image",
        "augmentation": AUGMENTATION,
        "mask_ext": ".png",
        "lazy": true
};

local BASE_ITERATOR = {
  "type": "basic",
  "batch_size": 4,
};

{
  "dataset_reader": READER,
  "validation_dataset_reader": READER,
  "train_data_path": "allencv/tests/fixtures/data/image_mask_reader",
  "validation_data_path": "allencv/tests/fixtures/data/image_mask_reader",
  "model": {
    "type": "semantic_segmentation",
    "encoder": {
        "type": "feature_pyramid",
        "backbone": {
            "type": "resnet_encoder",
            "model_str": "resnet34",
            "pretrained": false,
            "requires_grad": true
        },
        "output_channels": 64
    },
    "decoder": {
        "type": "basic",
        "input_scales": [1, 2, 4, 8],
        "input_channels": 64,
        "output_channels": 32
    },
    "num_classes": 3,
    "batch_size_per_image": 500
  },
  "iterator": BASE_ITERATOR,
  "trainer": {
    "num_epochs": NUM_EPOCHS,
    "validation_metric": "+accuracy",
    "should_log_learning_rate": true,
    "cuda_device" : -1,
    "optimizer": {
      "type": "adam",
      "lr": 1e-4,
    }
  }
}