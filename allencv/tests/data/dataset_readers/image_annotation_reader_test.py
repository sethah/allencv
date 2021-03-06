# pylint: disable=no-self-use,invalid-name
from allencv.common.testing.test_case import AllenCvTestCase
from allencv.data.dataset_readers import ImageAnnotationReader
from allencv.data.fields import ImageField, BoundingBoxField, KeypointField

from allennlp.data import Vocabulary
from allennlp.data.fields import ListField, LabelField
from allennlp.data.iterators import BasicIterator


class TestImageAnnotationReader(AllenCvTestCase):

    def test_reader_output(self):
        reader = ImageAnnotationReader(num_keypoints=4, keypoints=True)
        instances = reader.read(self.FIXTURES_ROOT / "data" / "image_annotation")
        assert(isinstance(instances, list))
        assert len(instances) == 1
        fields = instances[0].fields
        assert isinstance(fields['image'], ImageField)
        assert isinstance(fields['boxes'], ListField)
        assert isinstance(fields['box_classes'], ListField)
        assert isinstance(fields['keypoint_positions'], ListField)
        assert isinstance(fields['boxes'].field_list[0], BoundingBoxField)
        assert isinstance(fields['box_classes'].field_list[0], LabelField)
        assert isinstance(fields['keypoint_positions'].field_list[0], KeypointField)
        vocab = Vocabulary.from_instances(instances)
        for inst in instances:
            inst.index_fields(vocab)
        assert instances[0].as_tensor_dict()['boxes'].shape[1] == 4

    def test_keypoints_off(self):
        reader = ImageAnnotationReader(num_keypoints=4, keypoints=False)
        instances = reader.read(self.FIXTURES_ROOT / "data" / "image_annotation")
        fields = instances[0].fields
        assert set(fields.keys()) == {'image', 'image_sizes', 'boxes', 'box_classes'}

    def test_missing_annotations(self):
        reader = ImageAnnotationReader(augmentation=[], annotation_dir='xyz')
        instances = reader.read(self.FIXTURES_ROOT / "data" / "image_annotation")
        assert(isinstance(instances, list))
        assert len(instances) == 1
        fields = instances[0].fields
        assert isinstance(fields['image'], ImageField)


