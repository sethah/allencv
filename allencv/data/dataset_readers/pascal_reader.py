import numpy as np
from overrides import overrides

from typing import Dict, Iterable, List, Tuple
import logging

from allencv.data.transforms.image_transform import ImageTransform
from allencv.data.fields.image_field import ImageField, BoundingBoxField

from allennlp.data.instance import Instance
from allennlp.data.fields import LabelField, Field, ArrayField, ListField
from allennlp.data.dataset_readers import DatasetReader

from allencv.data.dataset_readers.image_classification_directory import ImageDatasetReader
from maskrcnn_benchmark.data.datasets.voc import PascalVOCDataset

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



@DatasetReader.register("pascal")
class PascalReader(ImageDatasetReader):
    """
    Wrapper for a ``PascalVOCDataset``

    Parameters
    ----------
    image_set: ``str``
        The name of the Pascal VOC split to use, ('train', 'val', etc.)
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be read lazily.
    """
    def __init__(self,
                 augmentation: List[ImageTransform],
                 image_set: str,
                 lazy: bool = False) -> None:
        super(PascalReader, self).__init__(augmentation)
        self.lazy = lazy
        self.image_set = image_set

    def _read(self, file_path: str) -> Iterable[Instance]:
        base_dataset = PascalVOCDataset(file_path, split=self.image_set)
        for i in range(len(base_dataset)):
            img, boxlist, idx = base_dataset[i]
            label_classes = boxlist.get_field("labels").numpy().tolist()
            img, _, boxes = self.augment(np.array(img), boxes=[b.numpy() for b in boxlist.bbox])
            h, w, c = img.shape
            yield self.text_to_instance(img, (w, h), boxes, label_classes)

    @overrides
    def text_to_instance(self,
                         image: np.ndarray,
                         image_size: Tuple[int, int],
                         label_box: List[List[float]] = None,
                         label_class: List[str] = None) -> Instance:
        fields: Dict[str, Field] = {}
        fields['image'] = ImageField(image.transpose(2, 0, 1), channels_first=True)
        fields['image_sizes'] = ArrayField(np.array([image_size[0], image_size[1]]))
        if label_box is not None:
            box_fields = [BoundingBoxField(x) for x in label_box]
            fields['boxes'] = ListField(box_fields)
            fields['box_classes'] = ListField([LabelField(idx, skip_indexing=True)
                                               for idx in label_class])
        return Instance(fields)
