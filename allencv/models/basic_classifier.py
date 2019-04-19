from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy

from allencv.modules.im2im_encoders.im2im_encoder import Im2ImEncoder
from allencv.modules.im2vec_encoders.im2vec_encoder import Im2VecEncoder


@Model.register("basic_image_classifier")
class BasicImageClassifier(Model):
    """
    This ``Model`` implements a basic image classifier. Optionally encode the input image
    with a ``Im2ImEncoder`` and then encode the image with a ``Im2VecEncoder``. That output
    is passed to a linear classification layer, which projects into the label space.

    Parameters
    ----------
    vocab : ``Vocabulary``
    im2im_encoder : ``Im2ImEncoder``, optional (default=``None``)
        Optional image encoder layer for the input image.
    im2vec_encoder : ``Im2VecEncoder``
        Required Im2Vec encoder layer.
    dropout : ``float``, optional (default = ``None``)
        Dropout percentage to use.
    num_labels: ``int``, optional (default = ``None``)
        Number of labels to project to in classification layer. By default, the classification layer will
        project to the size of the vocabulary namespace corresponding to labels.
    label_namespace: ``str``, optional (default = "labels")
        Vocabulary namespace corresponding to labels. By default, we use the "labels" namespace.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        If provided, will be used to initialize the model parameters.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 im2vec_encoder: Im2VecEncoder,
                 im2im_encoder: Im2ImEncoder = None,
                 dropout: float = None,
                 num_labels: int = None,
                 label_namespace: str = "labels",
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:

        super(BasicImageClassifier, self).__init__(vocab)
        self._im2vec_encoder = im2vec_encoder
        if im2im_encoder:
            self._im2im_encoder = im2im_encoder
        else:
            self._im2im_encoder = None
        self._classifier_input_dim = self._im2vec_encoder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=label_namespace)
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,  # type: ignore
                image: torch.FloatTensor,
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ

        output = image
        if self._im2im_encoder:
            output = self._im2im_encoder(output)

        output = self._im2vec_encoder(output)

        if self._dropout:
            output = self._dropout(output)

        logits = self._classification_layer(output)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {'accuracy': self._accuracy.get_metric(reset)}
        return metrics
