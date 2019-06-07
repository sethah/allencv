import torch

from allencv.common.testing import AllenCvTestCase
from allencv.data.dataset_readers import ImageClassificationDirectory
from allencv.predictors import ImagePredictor
from allencv.models import BasicImageClassifier
from allencv.modules.im2im_encoders import FeedforwardEncoder
from allencv.modules.im2vec_encoders import FlattenEncoder

from allennlp.data import Vocabulary


class TestBasicImageClassifier(AllenCvTestCase):

    def test_predictor(self):
        reader = ImageClassificationDirectory()
        input_channels = 3
        num_layers = 2
        hidden_dim = 64
        im_height = 224
        im_width = 224
        batch_size = 1
        im = torch.randn(batch_size, input_channels, im_height, im_width)
        im2im_encoder = FeedforwardEncoder(input_channels, num_layers, hidden_dim,
                                            activations='relu', downsample=True)
        im2vec_encoder = FlattenEncoder(im2im_encoder.get_output_channels(),
                                        im_height // (2**num_layers), im_width // (2**num_layers))
        vocab = Vocabulary({'labels': {'cat': 1, 'dog': 1}})
        classifier = BasicImageClassifier(vocab, im2vec_encoder, im2im_encoder)
        predictor = ImagePredictor(classifier, reader)
        predicted = predictor.predict_json({'image': im.numpy().ravel().tolist(),
                                       'image_shape': [224, 224, 3]})

        assert set(predicted.keys()) == {'logits', 'probs', 'prediction', 'class'}
        assert len(predicted['logits'])
        assert predicted['class'] in {'cat', 'dog'}
