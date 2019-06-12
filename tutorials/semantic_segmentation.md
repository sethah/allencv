# Semantic Segmentation

Set up your environment.

```bash
export ALLENCV_DIR=path/to/allencv
cd $ALLENCV_DIR
mkdir data && cd data
```

### Semantic Segmentation


Download and split PascalVOC data

```bash
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
cd $ALLENCV_DIR
unset VOCDIR
```

Train a simple segmentation model

```bash
TRAIN_PATH=$ALLENCV_DIR/data/pascal_voc_segmentation/train \
VALIDATION_PATH=$ALLENCV_DIR/data/pascal_voc_segmentation/val  \
allennlp train $ALLENCV_DIR/training_config/semantic_segmentation.jsonnet \
--include-package allencv.data.dataset_readers \
--include-package allencv.modules.image_encoders \
--include-package allencv.models.semantic_segmentation \
--include-package allencv.modules.image_decoders \
-s models/semantic_segmentation0
```