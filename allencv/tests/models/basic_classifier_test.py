from allencv.common.testing import AllenCvTestCase, ModelTestCase
from allencv.data.dataset_readers import ImageClassificationDirectory
from allencv.models import BasicImageClassifier
from allencv.modules.im2im_encoders import FeedforwardEncoder
from allencv.modules.im2vec_encoders import FlattenEncoder


class TestBasicImageClassifier(ModelTestCase):

    def test_basic_experiment(self):
        data_directory = AllenCvTestCase.FIXTURES_ROOT / "data" / "image_classification"
        self.set_up_model(AllenCvTestCase.FIXTURES_ROOT / 'basic_image_classifier' / 'experiment.jsonnet',
                          data_directory)
        self.ensure_model_can_train_save_and_load(self.param_file)

