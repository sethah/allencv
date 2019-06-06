from allencv.common.testing import AllenCvTestCase, ModelTestCase
from allencv.data.dataset_readers import ImageAnnotationReader
from allencv.models.object_detection import RPN, KeypointRCNN
from allencv.modules.image_encoders import ResnetEncoder, FPN


class TestKeypointRCNN(ModelTestCase):

    def test_basic_experiment(self):
        data_directory = AllenCvTestCase.FIXTURES_ROOT / "data" / "image_annotation"
        self.set_up_model(AllenCvTestCase.FIXTURES_ROOT / 'object_detection' / 'faster_rcnn' / 'keypoint_rcnn.jsonnet',
                          data_directory)
        self.ensure_model_can_train_save_and_load(self.param_file, tolerance=1e-4)

    # def test_pretrained_experiment(self):
    #     data_directory = AllenCvTestCase.FIXTURES_ROOT / "data" / "image_annotation"
    #     self.set_up_model(AllenCvTestCase.FIXTURES_ROOT / 'object_detection' / 'faster_rcnn' / 'faster_rcnn_pretrained.jsonnet',
    #                       data_directory)
    #     self.ensure_model_can_train_save_and_load(self.param_file, tolerance=1e-4)

