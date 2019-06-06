local NUM_EPOCHS = 15;

local AUGMENTATION = [
            {
                "type": "resize",
                "height": 224,
                "width": 224
            }, {
                "type": "normalize"
            }
        ];
local TRAIN_READER = {
        "type": "image_classification_directory",
        "augmentation": AUGMENTATION,
        "lazy": true
};
local VALID_READER = {
        "type": "image_classification_directory",
        "augmentation": AUGMENTATION,
        "lazy": true
};

local BASE_ITERATOR = {
  "type": "basic",
  "batch_size": 32,
};

{
  "dataset_reader": TRAIN_READER,
  "validation_dataset_reader": VALID_READER,
  "train_data_path": "allencv/tests/fixtures/data/image_classification",
  "validation_data_path": "allencv/tests/fixtures/data/image_classification",

  "vocabulary": {
      "max_vocab_size": 50000,
      "min_count": {"tokens": 3}
  },
  "model": {
    "type": "basic_image_classifier",
    "im2im_encoder": {
       "type": "feedforward",
       "input_channels": 3,
       "num_layers": 2,
       "hidden_channels": [8, 16],
       "activations": "relu",
       "downsample": true
    },
    "im2vec_encoder": {
      "type": "flatten",
      "input_channels": 16,
      "input_height": 56,
      "input_width": 56
    }
  },
  "iterator": BASE_ITERATOR,
  "trainer": {
    "num_epochs": NUM_EPOCHS,
    "validation_metric": "+accuracy",
    "cuda_device" : -1,
    "optimizer": {
      "type": "adam",
      "lr": 1e-3,
    }
  }
}