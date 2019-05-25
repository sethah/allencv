# AllenCV

A computer vision library built on top of [PyTorch](https://github.com/pytorch/pytorch) and 
[AllenNLP](https://github.com/allenai/allennlp).

## Installation

```
pip install allencv
```

Currently, the project requires the [maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) library, which must be installed manually
by following the instructions [here](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md). 
This dependence may be removed in the future.


## Test it out

Preliminaries
```
export ALLENCV_DIR=path/to/allencv
cd $ALLENCV_DIR
mkdir data && cd data
```

### Classification

Download cats and dogs dataset
```
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
