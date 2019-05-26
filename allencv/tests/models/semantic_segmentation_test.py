from allencv.common.testing import AllenCvTestCase, ModelTestCase
from allencv.data.dataset_readers import PairedImageReader
from allencv.models import SemanticSegmentationModel
from allencv.modules.image_encoders import ResnetEncoder, FPN
from allencv.modules.image_decoders import BasicDecoder


class TestSemanticSegmentation(ModelTestCase):

    def test_basic_experiment(self):
        data_directory = AllenCvTestCase.FIXTURES_ROOT / "data" / "image_mask_reader"
        self.set_up_model(AllenCvTestCase.FIXTURES_ROOT / 'semantic_segmentation' / 'experiment.jsonnet',
                          data_directory)
        self.ensure_model_can_train_save_and_load(self.param_file)
