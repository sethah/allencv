# AllenCV

[![Build Status](https://travis-ci.com/sethah/allencv.svg?branch=master)](https://travis-ci.com/sethah/allencv)
[![codecov](https://codecov.io/gh/sethah/allencv/branch/master/graph/badge.svg)](https://codecov.io/gh/sethah/allencv)

A computer vision library built on top of [PyTorch](https://github.com/pytorch/pytorch) and 
[AllenNLP](https://github.com/allenai/allennlp).

## Quick Links

* [Tutorials](tutorials)
* [Continuous Build](https://travis-ci.com/sethah/allencv)

## Installation

```
pip install allencv
```

Currently, the project requires the [maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) library, which must be installed manually
by following the instructions [here](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md). 
This dependence may be removed in the future.

### Installing from source

Clone the git repository:

  ```bash
  git clone https://github.com/sethah/allencv.git
  ```

In a Python 3.6 virtual environment, run:

  ```bash
  pip install --editable .
  ```

This will make `allencv` available on your system but it will use the sources from the local clone
you made of the source repository.

## Running AllenCV

AllenCV follows all the same conventions and supports the same interfaces as AllenNLP. Read more
about the AllenNLP command line interface [here](https://github.com/allenai/allennlp#running-allennlp).

## Test it out

Preliminaries
```
export ALLENCV_DIR=path/to/allencv
cd $ALLENCV_DIR
mkdir data && cd data
```

### Semantic Segmentation


Download and split PascalVOC data
```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar xf VOCtrainval_06-Nov-2007.tar

export VOCDIR=$PWD/VOCdevkit/VOC2007
mkdir pascal_voc_segmentation && cd pascal_voc_segmentation
for DIR in 'train' 'val'
do
  mkdir $DIR
  mkdir $DIR/masks
  mkdir $DIR/images
  ls $VOCDIR/SegmentationClass | shuf -n 200 | xargs -i mv $VOCDIR/SegmentationClass/{} $DIR/masks/
  ls $DIR/masks | sed -e 's/\.png$/\.jpg/' | xargs -i mv $VOCDIR/JPEGImages/{} $DIR/images/
done
cd ..
unset VOCDIR
```

Train a simple segmentation model

```
TRAIN_PATH=$ALLENCV_DIR/data/pascal_voc_segmentation/train \
VALIDATION_PATH=$ALLENCV_DIR/data/pascal_voc_segmentation/val  \
allennlp train $ALLENCV_DIR/training_config/semantic_segmentation.jsonnet \
--include-package allencv.data.dataset_readers \
--include-package allencv.modules.image_encoders \
--include-package allencv.models.semantic_segmentation \
--include-package allencv.modules.image_decoders \
-s models/semantic_segmentation0
```
