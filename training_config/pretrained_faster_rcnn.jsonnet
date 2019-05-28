local NUM_GPUS = 1;
local NUM_EPOCHS = 40;

local AUGMENTATION = [
            {
                "type": "resize",
                "height": 512,
                "width": 512
            }, {
                "type": "bgr_normalize"
            }
        ];
local READER = {
        "type": "image_annotation",
        "augmentation": AUGMENTATION,
        "lazy": true
};

local BASE_ITERATOR = {
  "type": "basic",
  "batch_size": 2 * NUM_GPUS,
};

local MODEL = {
    "type": "pretrained_detectron_faster_rcnn",
    "rpn": {
        "type": "detectron_rpn",
        "anchor_sizes": [64, 128, 256, 512],
        "anchor_strides": [4, 8, 16, 32]
    },
    "decoder_thresh": 0.1
};

{
  "dataset_reader": READER,
  "train_data_path": std.extVar("TRAIN_PATH"),
  "model": MODEL,
  "iterator": BASE_ITERATOR,
  "trainer": {
    "type": "no_op",
  }
}