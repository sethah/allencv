# pylint: disable=no-self-use,invalid-name
from allencv.common.testing.test_case import AllenCvTestCase
from allencv.data.dataset_readers import ImageClassificationDirectory
from allencv.data.fields import ImageField

from allennlp.data.fields import LabelField


class TestImageClassificationDirectoryReader(AllenCvTestCase):

    def test_reader_output(self):
        reader = ImageClassificationDirectory([])
        instances = reader.read(self.FIXTURES_ROOT / "data" / "image_classification")
        assert(isinstance(instances, list))
        assert len(instances) == 2
        fields = instances[0].fields
        assert isinstance(fields['image'], ImageField)
        assert isinstance(fields['label'], LabelField)
        assert set([inst.fields['label'].label for inst in instances]) == {'Cat', 'Dog'}


