# Image classification

This document describes how to train a basic image classification model using AllenCV.

AllenCV provides a `BasicImageClassifier` model out-of-the-box, which makes it simple to train
a classifier. It relies on two underlying abstractions: `Im2ImEncoder` and `Im2VecEncoder`. An
`Im2ImEncoder` is a neural network that transforms one input image to another image which
contains higher level features. One example is the convolutional seciont of a ResNet model,
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


## Training data

Set up your environment.

```bash
export ALLENCV_DIR=path/to/allencv
cd $ALLENCV_DIR
mkdir data && cd data
```

Download the cats and dogs dataset.

```bash
wget https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip
unzip kagglecatsanddogs_3367a.zip
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

Build a simple classifier

```
TRAIN_PATH=$ALLENCV_DIR/data/PetImages/train \
VALIDATION_PATH=$ALLENCV_DIR/data/PetImages/valid \
allennlp train $ALLENCV_DIR/training_config/image_classifier.jsonnet \
--include-package allencv.data.dataset_readers \
--include-package allencv.modules.im2im_encoders \
--include-package allencv.models.basic_classifier \
--include-package allencv.modules.im2vec_encoders \
-s path/to/serialization/dir
```