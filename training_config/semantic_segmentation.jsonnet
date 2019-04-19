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
        "type": "segmentation_reader",
        "augmentation": TRAIN_AUGMENTATION,
        "lazy": true
};
local VALID_READER = {
        "type": "segmentation_reader",
        "augmentation": TRAIN_AUGMENTATION,
        "lazy": true
};

local BASE_ITERATOR = {
  "type": "basic",
  "max_instances_in_memory": 16384 * NUM_GPUS,
  "batch_size": 8 * NUM_GPUS,
};

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
            "type": "pretrained_resnet",
            "resnet_model": "resnet34",
            "requires_grad": true
        },
        "output_channels": 128
    },
    "decoder": {
        "type": "basic",
        "input_scales": [1, 2, 4, 8],
        "input_channels": 128,
        "output_channels": 256
    },
    "num_classes": 2
  },
  "iterator": BASE_ITERATOR,
  "trainer": {
    "num_epochs": NUM_EPOCHS,
    "validation_metric": "+accuracy",
    "should_log_learning_rate": true,
    "cuda_device" : if NUM_GPUS > 1 then std.range(0, NUM_GPUS - 1) else 0,
    "optimizer": {
      "type": "adam",
      "lr": 1e-3,
      "parameter_groups": [
      [["^_encoder\\._backbone\\.stages\\.0"], {"initial_lr": 0.0005}],
      [["^_encoder\\._backbone\\.stages\\.1"], {"initial_lr": 0.0005}],
      [["^_encoder\\._backbone\\.stages\\.2"], {"initial_lr": 0.001}],
      [["^_encoder\\._backbone\\.stages\\.3"], {"initial_lr": 0.001}],
      [["(^_encoder\\._combine|^_decoder|^_encoder\\._convert|^upsampling|^final_conv)"], {"initial_lr": 0.001}],
     ]
    },
    "learning_rate_scheduler": {
        "type": "slanted_triangular",
        "num_epochs": NUM_EPOCHS,
        "num_steps_per_epoch": 130,
        "gradual_unfreezing": true,
        "discriminative_fine_tuning": true
    },
    "patience": 20
  }
}