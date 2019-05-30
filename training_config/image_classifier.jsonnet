local NUM_GPUS = 1;
local NUM_EPOCHS = 5;

local TRAIN_AUGMENTATION = [
            {
                "type": "resize",
                "height": 224,
                "width": 224
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
                "height": 224,
                "width": 224
            }, {
                "type": "normalize",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        ];

local TRAIN_READER = {
        "type": "image_classification_directory",
        "augmentation": TRAIN_AUGMENTATION,
        "lazy": true
};
local VALID_READER = {
        "type": "image_classification_directory",
        "augmentation": VALID_AUGMENTATION,
        "lazy": true
};

local BASE_ITERATOR = {
  "type": "basic",
  "batch_size": 16 * NUM_GPUS,
};

{
  "dataset_reader": TRAIN_READER,
  "validation_dataset_reader": VALID_READER,
  "train_data_path": std.extVar("TRAIN_PATH"),
  "validation_data_path": std.extVar("VALIDATION_PATH"),

  "model": {
    "type": "basic_image_classifier",
    "im2im_encoder": {
       "type": "pretrained_resnet",
       "resnet_model": "resnet50",
       "requires_grad": true
    },
    "im2vec_encoder": {
      "type": "flatten",
      "input_channels": 2048,
      "input_height": 1,
      "input_width": 1
    }
  },
  "iterator": BASE_ITERATOR,
  "trainer": {
    "num_epochs": NUM_EPOCHS,
    "validation_metric": "+accuracy",
    "should_log_learning_rate": true,
    "cuda_device" : if NUM_GPUS > 1 then std.range(0, NUM_GPUS - 1) else 0,
    "optimizer": {
      "type": "adam",
      "lr": 1e-5,
      "parameter_groups": [
          [["^_im2im_encoder\\.model\\.(0|1)"], {"initial_lr": 1e-5}],
          [["^_im2im_encoder\\.model\\.4"], {"initial_lr": 1e-5}],
          [["^_im2im_encoder\\.model\\.5"], {"initial_lr": 1e-5}],
          [["^_im2im_encoder\\.model\\.6"], {"initial_lr": 1e-5}],
          [["^_im2im_encoder\\.model\\.7"], {"initial_lr": 1e-5}],
          [["^_classification_layer"], {"initial_lr": 1e-5}]
     ]
    },
    "learning_rate_scheduler": {
        "type": "slanted_triangular",
        "num_epochs": NUM_EPOCHS,
        "num_steps_per_epoch": 260,
        "gradual_unfreezing": true,
        "discriminative_fine_tuning": true
    },
    "patience": 20
  }
}