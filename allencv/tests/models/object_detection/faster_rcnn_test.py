import torch
import torch.nn as nn

from allennlp.data import Instance

from allencv.common.testing import ModelTestCase
from allencv.models.object_detection import RPN, RCNN, PretrainedDetectronFasterRCNN, FasterRCNN
from allencv.modules.image_encoders import ResnetEncoder, FPN
from allencv.modules.im2vec_encoders import FlattenEncoder

from allencv.common.testing import AllenCvTestCase, ModelTestCase
from allencv.data.dataset_readers import ImageAnnotationReader
from allencv.models.object_detection import RPN, PretrainedDetectronRPN
from allencv.modules.image_encoders import ResnetEncoder, FPN

from allennlp.modules import FeedForward


class TestFasterRCNN(ModelTestCase):

    def test_basic_experiment(self):
        data_directory = AllenCvTestCase.FIXTURES_ROOT / "data" / "image_annotation"
        self.set_up_model(AllenCvTestCase.FIXTURES_ROOT / 'object_detection' / 'faster_rcnn' / 'faster_rcnn.jsonnet',
                          data_directory)
        self.ensure_model_can_train_save_and_load(self.param_file, tolerance=1e-4)

    def test_pretrained_experiment(self):
        data_directory = AllenCvTestCase.FIXTURES_ROOT / "data" / "image_annotation"
        self.set_up_model(AllenCvTestCase.FIXTURES_ROOT / 'object_detection' / 'faster_rcnn' / 'faster_rcnn_pretrained.jsonnet',
                          data_directory)
        self.ensure_model_can_train_save_and_load(self.param_file, tolerance=1e-4)

    def test_forward(self):
        rpn = PretrainedDetectronRPN()
        model = PretrainedDetectronFasterRCNN(rpn)
        reader = ImageAnnotationReader()
        instances = reader.read(AllenCvTestCase.FIXTURES_ROOT / "data" / "image_annotation")
        model.eval()
        inst2 = Instance({k: v for k, v in instances[0].fields.items() if k not in {'boxes', 'box_classes', 'keypoint_positions'}})
        print(inst2)
        out = model.forward_on_instance(instances[0])
        out2 = model.forward_on_instance(inst2)

