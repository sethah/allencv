from allennlp.common import Registrable


class ImageTransform(Registrable):

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, **kwargs):
        self.transform(**kwargs)
