# pylint: disable=no-self-use,invalid-name
from allencv.common.testing.test_case import AllenCvTestCase
from allencv.data.dataset_readers import ImageAnnotationReader
from allencv.data.fields import ImageField, BoundingBoxField

from allennlp.data.fields import ListField


class TestImageAnnotationReader(AllenCvTestCase):

    def test_reader_output(self):
        reader = ImageAnnotationReader()
        instances = reader.read(self.FIXTURES_ROOT / "data" / "image_annotation")
        assert(isinstance(instances, list))
        assert len(instances) == 1
        fields = instances[0].fields
        assert isinstance(fields['image'], ImageField)
        assert isinstance(fields['boxes'], ListField)
        assert isinstance(fields['boxes'].field_list[0], BoundingBoxField)

    def test_missing_annotations(self):
        reader = ImageAnnotationReader(augmentation=[], annotation_dir='xyz')
        instances = reader.read(self.FIXTURES_ROOT / "data" / "image_annotation")
        assert(isinstance(instances, list))
        assert len(instances) == 1
        fields = instances[0].fields
        assert isinstance(fields['image'], ImageField)


