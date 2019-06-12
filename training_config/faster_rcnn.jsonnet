local NUM_GPUS = 1;
local NUM_THREADS = 1;
local NUM_EPOCHS = 40;

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
    }
};

local MODEL = {
    "type": "faster_rcnn",
    "rpn": RPN,
    "train_rpn": true,
    "roi_box_head": {
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
        },
        "decoder_thresh": 0.05,
        "decoder_nms_thresh": 0.2
    }
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
      "lr": 5e-2,
      "parameter_groups": [
      [["^rpn\\.backbone\\._backbone\\.stages\\.0\\."], {"initial_lr": 0.00001}],
      [["^rpn\\.backbone\\._backbone\\.stages\\.1\\."], {"initial_lr": 0.0001}],
      [["^rpn\\.backbone\\._backbone\\.stages\\.2\\."], {"initial_lr": 0.0003}],
      [["^rpn\\.backbone\\._backbone\\.stages\\.3\\."], {"initial_lr": 0.0005}],
      [["(^cls_score)|(^bbox_pred)|(^feature_extractor)|(^rpn\\.backbone\\._convert)|(^rpn\\.backbone\\._combine)|(^rpn\\.conv)|(^rpn\\.cls_logits)|(^rpn\\.bbox_pred)"], {"initial_lr": 0.005}],
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