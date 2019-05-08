# pylint: disable=no-self-use,invalid-name
from allencv.common.testing.test_case import AllenCvTestCase
from allencv.data.dataset_readers import PairedImageReader
from allencv.data.fields import ImageField, MaskField


class TestImageMaskReader(AllenCvTestCase):

    def test_reader_output(self):
        reader = PairedImageReader(mask_ext=".png")
        instances = reader.read(TestImageMaskReader.FIXTURES_ROOT / "data" / "image_mask_reader")
        assert(isinstance(instances, list))
        assert len(instances) == 1
        fields = instances[0].fields
        assert isinstance(fields['image'], ImageField)
        assert isinstance(fields['mask'], MaskField)

    def test_lazy(self):
        reader = PairedImageReader(mask_ext=".png", lazy=True)
        instances = reader.read(TestImageMaskReader.FIXTURES_ROOT / "data" / "image_mask_reader")
        assert not isinstance(instances, list)
        assert len(list(instances)) == 1
        fields = list(instances)[0].fields
        assert isinstance(fields['image'], ImageField)
        assert isinstance(fields['mask'], MaskField)

    def test_mask_extension_default(self):
        reader = PairedImageReader(mask_dir="images")
        instances = reader.read(TestImageMaskReader.FIXTURES_ROOT / "data" / "image_mask_reader")
        assert len(instances) == 1


