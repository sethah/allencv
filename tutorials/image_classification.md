# Image classification

This document describes how to train a basic image classification model using AllenCV

## Data downloading

For this tutorial, we're going to learn to classify the classic Cats and Dogs dataset.
Download the cats and dogs dataset.

```bash
export ALLENCV_DIR=path/to/allencv
cd $ALLENCV_DIR
mkdir data && cd data

wget https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip
unzip -q kagglecatsanddogs_3367a.zip
cd PetImages
for DIR in 'train' 'val'
do
  mkdir $DIR
  for label in 'Cat' 'Dog'
  do
    mkdir $DIR/$label
    ls ./$label | shuf -n 1000 | xargs -i mv ./$label/{} $DIR/$label
  done
done
```

## Data loading

AllenCV comes with several built-in dataset readers that take care of reading images and 
performing data augmentation. In this case we'll use the `ImageClassificationDirectory`
dataset, which reads images separated into classes by their top-level directory. Our data 
should be structured like so:

```
train
    |-- Cat
    |   |-- 00001.jpg
    |   |-- 00002.jpg
    |   |-- ...
    |-- Dog
    |   |-- 00003.jpg
    |   |-- 00004.jpg
    |   |-- ...
valid
    |-- Cat
    |   |-- 000017.jpg
    |   |-- 000201.jpg
    |   |-- ...
    |-- Dog
    |   |-- 000034.jpg
    |   |-- 000118.jpg
    |   |-- ...
```

Each dataset inherits from a common `ImageDatasetReader` which is responsible for applying
image augmentation to each image. We can configure the dataset reader as follows:

```
"dataset_reader": {
    "type": "image_classification_directory",
    "augmentation": [
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
    ],
    "lazy": true
};
```

Defining image augmentation is simple using this dataset reader. The underlying implementation is
 based on the [Albumentations]() library. For image classification augmentation is only applied to
 the input images, but in more complex cases dealing with bounding boxes or pixel masks augmentation
 will need to be applied to labels as well. This is automatically handled by the dataset readers.

In general, some augmentation should not be applied to the validation data since we will want
to emulate the deployment environment. It is easiest to do this by defining a validation
reader with a subset of the train augmentation.

```
"validation_dataset_reader": {
    "type": "image_classification_directory",
    "augmentation": [
        {
            "type": "resize",
            "height": 512,
            "width": 512
        }, {
            "type": "normalize",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    ],
    "lazy": true
};
```

## Training

AllenCV provides a `BasicImageClassifier` model out-of-the-box, which makes it simple to train
a classifier. It relies on two underlying abstractions: `Im2ImEncoder` and `Im2VecEncoder`. An
`Im2ImEncoder` is a neural network that transforms one input image to another image which
contains higher level features. One example is the convolutional section of a ResNet model,
which outputs a `1024x7x7` image of features, for example. An `Im2VecEncoder` would typically
take a featurized input image and convert that to a fixed length vector that can be used
for classification.

A basic classifier might be defined interactively as follows:

```python
vocab = Vocabulary({'labels': {'Cat': 1234, 'Dog': 1234}})
im2im_encoder = ResnetEncoder('resnet50')
im2vec_encoder = FlattenEncoder(input_channels=2048, input_height=1, input_width=1)
classifier = BasicImageClassifier(vocab, im2vec_encoder, im2im_encoder)
```

The same classifier could be defined declaratively using a config file.

```
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
}
```

Following the AllenNLP conventions, we'll need to define a few other components, including
a trainer. When using pre-trained models, it will often be advantageous to continue fine-tuning
the model. However, to prevent the model from "forgetting" what it already knows. We can easily
implement gradual unfreezing using the `SlantedTriangular` learning rate scheduler. This can
be defined as follows:

```
"trainer": {
    "num_epochs": NUM_EPOCHS,
    "validation_metric": "+accuracy",
    "cuda_device" : 0,
    "optimizer": {
      "type": "adam",
      "lr": 1e-3
      "parameter_groups": [
          [["^_im2im_encoder\\.model\\.(0|1)"]],
          [["^_im2im_encoder\\.model\\.4"]],
          [["^_im2im_encoder\\.model\\.5"]],
          [["^_im2im_encoder\\.model\\.6"]],
          [["^_im2im_encoder\\.model\\.7"]],
          [["^_classification_layer"]]
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
```

With data loading, model, and trainer defined we can now build a simple classifier

```
TRAIN_PATH=$ALLENCV_DIR/data/PetImages/train \
VALIDATION_PATH=$ALLENCV_DIR/data/PetImages/val \
allennlp train $ALLENCV_DIR/training_config/image_classifier.jsonnet \
--include-package allencv.data.dataset_readers \
--include-package allencv.modules.im2im_encoders \
--include-package allencv.models.basic_classifier \
--include-package allencv.modules.im2vec_encoders \
-s models/catsanddogs 
```

# Predictions

We'll use the the AllenNLP `Predictor` abstraction to make json to json predictions. First, 
we need an input.

```
echo '{"image_path": "/path/to/test/image.jpg"}' > test_input.json
```

We can do this use the `Predictor` api programmatically:

```python
archive = load_archive('/path/to/model.tar.gz')
predictor = Predictor.from_archive(archive, 'default_image')
predictor = ImagePredictor(model, reader)
predictor.predict({'image_path': "/path/to/test/image.jpg"})
```

Or use the more standard `allennlp predict` command. 

```
allennlp predict models/catsanddogs/model.tar.gz test_input.json \
--cuda-device 0 --predictor default_image \
--include-package allencv.data.dataset_readers \
--include-package allencv.modules.im2vec_encoders \
--include-package allencv.modules.im2im_encoders \
--include-package allencv.models \
--include-package allencv.predictors
```

## Running a Web Demo

Once you have a trained model and a predictor,
it's easy to run a simple web demo:

```
python -m allencv.service.server_simple \
    --archive-path models/catsanddogs/model.tar.gz \
    --predictor default_image \
    --include-package allencv.data.dataset_readers \
    --include-package allencv.modules.im2vec_encoders \
    --include-package allencv.modules.im2im_encoders \
    --include-package allencv.models \
    --include-package allencv.predictors \
    --title "Cats and Dogs Classifier" \
    --classification
```

This will start a server at `localhost:8000` that just allows simple
visualizations of predictions.
