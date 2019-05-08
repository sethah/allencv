local NUM_GPUS = 1;
local NUM_THREADS = 1;
local NUM_EPOCHS = 15;

local TRAIN_AUGMENTATION = [
            {
                "type": "resize",
                "height": 512,
                "width": 512
            }, {
                "type": "normalize",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
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
                "height": 512,
                "width": 512
            }, {
                "type": "normalize",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
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
        "image_set": "val",
        "lazy": true
};

local BASE_ITERATOR = {
  "type": "basic",
  "batch_size": 2 * NUM_GPUS,
};

local RPN = {
    "backbone": {
        "type": "feature_pyramid",
        "backbone": {
            "type": "pretrained_resnet",
            "resnet_model": "resnet50",
            "requires_grad": true
        },
        "output_channels": 256
    },
    "anchor_sizes": [64, 128, 256, 512],
    "anchor_strides": [4, 8, 16, 32]
};

local MODEL = {
    "type": "faster_rcnn",
    "rpn": RPN,
    "roi_feature_extractor": {
        "type": "flatten",
        "input_channels": 256,
        "input_height": 7,
        "input_width": 7,
        "feedforward": {
            "input_dim": 7*7*256,
            "num_layers": 2,
            "hidden_dims": [1024, 512],
            "activations": 'relu'
        }
    },
    num_labels: 21
};

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
      "type": "adam",
      "lr": 1e-3,
      "parameter_groups": [
      [["^rpn\\.backbone\\._backbone\\.stages\\.0\\."], {"initial_lr": 0.000001}],
      [["^rpn\\.backbone\\._backbone\\.stages\\.1\\."], {"initial_lr": 0.00001}],
      [["^rpn\\.backbone\\._backbone\\.stages\\.2\\."], {"initial_lr": 0.00003}],
      [["^rpn\\.backbone\\._backbone\\.stages\\.3\\."], {"initial_lr": 0.00005}],
      [["(^rpn\\.backbone\\._convert)|(^rpn\\.backbone\\._combine)|(^rpn\\.conv)|(^rpn\\.cls_logits)|(^rpn\\.bbox_pred)"], {"initial_lr": 0.0001}],
      [["(^cls_score)|(^bbox_pred)|(^feature_extractor)"], {"initial_lr": 0.001}],
     ]
    },
    "learning_rate_scheduler": {
        "type": "slanted_triangular",
        "num_epochs": NUM_EPOCHS,
        "num_steps_per_epoch": 1250,
        "gradual_unfreezing": true,
        "discriminative_fine_tuning": true
    },
    "patience": 20
  }
}