import numpy as np
from overrides import overrides
from PIL import Image

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor


@Predictor.register("default_image")
class ImagePredictor(Predictor):

    def predict(self, image_path: str) -> JsonDict:
        return self.predict_json({"image_path": image_path})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        if "image_path" in json_dict:
            image_path = json_dict["image_path"]
            img = np.array(Image.open(image_path).convert('RGB'))
        elif "image" in json_dict:
            img = np.array(json_dict['image']).reshape(json_dict['image_shape']).astype(np.float64)
        return self._dataset_reader.text_to_instance(img)
