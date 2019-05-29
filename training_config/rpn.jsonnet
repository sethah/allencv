local NUM_GPUS = 1;
local NUM_THREADS = 1;
local NUM_EPOCHS = 30;

local TRAIN_AUGMENTATION = [
            {
                "type": "resize",
                "height": 512,
                "width": 512
            }, {
                "type": "normalize"
            }, {
                "type": "horizontal_flip",
                "p": 0.5
            }
        ];
local VALID_AUGMENTATION = [
            {
                "type": "resize",
                "height": 512,
                "width": 512
            }, {
                "type": "normalize"
            }
        ];
local TRAIN_READER = {
        "type": "pascal",
        "augmentation": TRAIN_AUGMENTATION,
        "image_set": "train",
        "lazy": true
};
local VALID_READER = {
        "type": "pascal",
        "augmentation": VALID_AUGMENTATION,
        "image_set": "val_small",
        "lazy": true
};

local BASE_ITERATOR = {
  "type": "basic",
  "batch_size": 2 * NUM_GPUS,
};

local MODEL = {
    "type": "rpn",
    "backbone": {
        "type": "feature_pyramid",
        "backbone": {
            "type": "resnet_encoder",
            "resnet_model": "resnet101",
            "pretrained": "true",
            "requires_grad": "true"
        },
        "output_channels": 256
    },
    "anchor_sizes": [[32], [64], [128], [256], [512]],
    "anchor_aspect_ratios": [[0.5, 1.0, 2.0], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0]]
};

local start_momentum = 0.9;
{
  "dataset_reader": TRAIN_READER,
  "validation_dataset_reader": VALID_READER,
  "train_data_path": std.extVar("TRAIN_PATH"),
  "validation_data_path": std.extVar("VALIDATION_PATH"),
  "model": MODEL,
  "iterator": BASE_ITERATOR,
  "trainer": {
    "num_epochs": NUM_EPOCHS,
    "should_log_learning_rate": true,
    "cuda_device" : if NUM_GPUS > 1 then std.range(0, NUM_GPUS - 1) else 0,
    "optimizer": {
      "type": "sgd",
      "lr": 1e-2,
      "momentum": start_momentum,
      "parameter_groups": [
      [["^backbone\\._backbone\\.stages\\.0\\."], {"initial_lr": 1e-3, "momentum": start_momentum}],
      [["^backbone\\._backbone\\.stages\\.1\\."], {"initial_lr": 1e-3, "momentum": start_momentum}],
      [["^backbone\\._backbone\\.stages\\.2\\."], {"initial_lr": 1e-3, "momentum": start_momentum}],
      [["^backbone\\._backbone\\.stages\\.3\\."], {"initial_lr": 1e-3, "momentum": start_momentum}],
      [["(^backbone\\._convert)|(^backbone\\._combine)|(^conv)|(^cls_logits)|(^bbox_pred)"], {"initial_lr": 1e-3, "momentum": start_momentum}],
     ]
    },
    "learning_rate_scheduler": {
        "type": "slanted_triangular",
        "num_epochs": NUM_EPOCHS,
        "num_steps_per_epoch": 1250,
        "discriminative_fine_tuning": true,
        "gradual_unfreezing": true,
        "cut_frac": 0.3
    },
    "momentum_scheduler": {
        "type": "inverted_triangular",
        "cool_down": 10,
        "warm_up": 20,
    },
    "patience": 20
  }
}