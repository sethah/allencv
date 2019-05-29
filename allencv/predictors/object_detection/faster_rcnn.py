import numpy as np
from overrides import overrides
from PIL import Image

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor


@Predictor.register("faster_rcnn")
class FasterRCNNPredictor(Predictor):

    def predict(self, image_path: str) -> JsonDict:
        return self.predict_json({"image_path": image_path})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        image_path = json_dict["image_path"]
        img = Image.open(image_path).convert('RGB')
        return self._dataset_reader.text_to_instance(np.array(img))
