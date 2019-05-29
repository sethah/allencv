local NUM_GPUS = 1;
local NUM_THREADS = 1;
local NUM_EPOCHS = 30;

local image_height = 224;
local image_width = 224;
local TRAIN_AUGMENTATION = [
            {
                "type": "resize",
                "height": image_height,
                "width": image_width
            }, {
                "type": "normalize"
            }, {
                "type": "flip",
                "p": 0.5
            }, {
                "type": "channel_shuffle",
                "p": 0.5
            }
        ];
local VALID_AUGMENTATION = [
            {
                "type": "resize",
                "height": image_height,
                "width": image_width
            }, {
                "type": "normalize"
            }
        ];
local TRAIN_READER = {
        "type": "paired_image",
        "augmentation": TRAIN_AUGMENTATION,
        "mask_ext": ".png",
        "lazy": true
};
local VALID_READER = {
        "type": "paired_image",
        "augmentation": VALID_AUGMENTATION,
        "mask_ext": ".png",
        "lazy": true
};

local BASE_ITERATOR = {
  "type": "basic",
  "max_instances_in_memory": 16384 * NUM_GPUS,
  "batch_size": 16 * NUM_GPUS,
};

local initial_lr = 5e-3;
{
  "dataset_reader": TRAIN_READER,
  "validation_dataset_reader": VALID_READER,
  "train_data_path": std.extVar("TRAIN_PATH"),
  "validation_data_path": std.extVar("VALIDATION_PATH"),
  "model": {
    "type": "semantic_segmentation",
    "encoder": {
        "type": "feature_pyramid",
        "backbone": {
            "type": "resnet_encoder",
            "resnet_model": "resnet101",
            "pretrained": true,
            "requires_grad": true
        },
        "output_channels": 256
    },
    "decoder": {
        "type": "basic",
        "input_scales": [1, 2, 4, 8],
        "input_channels": 256,
        "output_channels": 128
    },
    "num_classes": 21,
    "batch_size_per_image": 30000
  },
  "iterator": BASE_ITERATOR,
  "trainer": {
    "num_epochs": NUM_EPOCHS,
    "validation_metric": "+accuracy",
    "should_log_learning_rate": true,
    "cuda_device" : if NUM_GPUS > 1 then std.range(0, NUM_GPUS - 1) else 0,
    "optimizer": {
      "type": "adam",
      "lr": initial_lr
    },
    "learning_rate_scheduler": {
        "type": "slanted_triangular",
        "num_epochs": NUM_EPOCHS,
        "num_steps_per_epoch": 160,
        "decay_factor": 0.25
    },
    "summary_interval": 20,
    "patience": 20
  }
}