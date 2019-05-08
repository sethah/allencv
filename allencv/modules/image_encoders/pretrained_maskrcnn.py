# @ImageEncoder.register("pretrained_resnet")
# class PretrainedResnetEncoder(ResnetEncoder):
#     """
#     Parameters
#     ----------
#     resnet_model: ``str``
#         Name of the pretrained Resnet variant.
#     requires_grad: ``bool``, optional (default = ``False``)
#         Whether to continue training the Resnet model.
#     num_layers: ``int``, optional (default = ``4``)
#         How many of the 4 Resnet layers to include in the encoder.
#     """
#     def __init__(self, model_str: str, requires_grad: bool = False):
#
#
#         for param in model.parameters():
#             param.requires_grad = requires_grad
#
#         super(PretrainedResnetEncoder, self).__init__(model)