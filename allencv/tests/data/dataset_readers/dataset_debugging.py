# pylint: disable=no-self-use,invalid-name
from pathlib import Path
import pytest
from unittest import TestCase

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image

from allencv.data.dataset_readers import ImageClassificationDirectory
from allencv.data.fields import ImageField, MaskField
from allencv.data.transforms.image_transform import ImageTransform
from allennlp.data.dataset_readers import MultiprocessDatasetReader
from allennlp.data.iterators import *
from allennlp.data import Vocabulary
import time

# import albumentations as aug

class MyDataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """
    def __init__(self, dir: str, transform):
        self.dir = Path(dir)
        self.files = list(self.dir.iterdir())
        self.transform = transform

    def __getitem__(self, index):
        file = self.files[index]
        img = Image.open(file)
        sample = img.convert("RGB")

        return self.transform(sample)

    def __len__(self):
        return len(self.files)

class TestImageMaskReader(TestCase):

    PROJECT_ROOT = (Path(__file__).parent / ".." / ".." / ".." / "..").resolve()  # pylint: disable=no-member
    MODULE_ROOT = PROJECT_ROOT / "allencv"
    TESTS_ROOT = MODULE_ROOT / "tests"
    FIXTURES_ROOT = TESTS_ROOT / "fixtures"

    # def test_reader_output(self):
    #     tfm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
    #                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    #     dataset = MyDataset("/Users/shendrickson/Development/allencv/data/processed/PetImagesSmall/Cat", tfm)
    #     loader = DataLoader(dataset, batch_size=8, num_workers=4)
    #     for batch in loader:
    #         print(batch.shape)

    # def test_allen(self):
    #     tfm = [ImageTransform(aug.Resize(224, 224)), ImageTransform(aug.Normalize())]
    #     reader = ImageClassificationDirectory(tfm)
    #     d = "/Users/shendrickson/Development/allencv/data/processed/PetImages/"
    #     t0 = time.time()
    #     instances = reader.read(d)
    #     base_iterator = BasicIterator(batch_size=8)
    #     vocab = Vocabulary.from_instances(instances)
    #     iterator = MultiprocessIterator(base_iterator=base_iterator, num_workers=3)
    #     iterator.index_with(vocab)
    #     for batch in iterator(instances, num_epochs=1):
    #         shp = batch['image'].shape
    #     t1 = time.time()
    #     print("-" * 20)
    #     print(t1 - t0)
    #     print("-" * 20)

    # def test_allen_multi(self):
    #     tfm = [ImageTransform(aug.Resize(224, 224)), ImageTransform(aug.Normalize())]
    #     reader = ImageClassificationDirectory(tfm)
    #     d = "/Users/shendrickson/Development/allencv/data/processed/PetImages2/"
    #     t0 = time.time()
    #     instances = reader.read(d)
    #     base_iterator = BasicIterator(batch_size=8)
    #     vocab = Vocabulary.from_instances(instances)
    #     base_iterator.index_with(vocab)
    #     for batch in base_iterator(instances, num_epochs=1):
    #         shp = batch['image'].shape
    #     t1 = time.time()
    #     print("-" * 20)
    #     print(t1 - t0)
    #     print("-" * 20)

    def test_allen_multi_reader(self):
        # tfm = [ImageTransform(aug.Resize(224, 224)), ImageTransform(aug.Normalize())]
        import sys, os
        fname = str(os.getpid()) + ".out"
        # sys.stdout = open(str(Path("/tmp/") / fname), "a")
        base_reader = ImageClassificationDirectory([])
        reader = MultiprocessDatasetReader(base_reader, num_workers=1)
        # reader = base_reader
        d = "/Users/shendrickson/Development/allencv/data/processed/PetImagesMulti2/*"
        t0 = time.time()
        instances = reader.read(d)
        print(len(list(instances)))
        # base_iterator = BasicIterator(batch_size=8)
        # vocab = Vocabulary.from_instances(instances)
        # base_iterator.index_with(vocab)
        # for batch in base_iterator(instances, num_epochs=1):
        #     shp = batch['image'].shape
        t1 = time.time()
        print("-" * 20)
        print(t1 - t0)
        print("-" * 20)

