# AllenCV

A computer vision library proof-of-concept that mimicks the interfaces 
in [AllenNLP](https://github.com/allenai/allennlp).

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
VALIDATION_PATH=$ALLENCV_DIR/allencv/data/PetImages/valid \
allennlp train $ALLENCV_DIR/allencv/training_config/image_classifier.jsonnet \
--include-package allencv.data.dataset_readers \
--include-package allencv.modules.im2im_encoders \
--include-package allencv.models.basic_classifier \
--include-package allencv.modules.im2vec_encoders \
-s path/to/serialization/dir
```
