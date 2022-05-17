from model.resnet import CategoricalNetwork
from model.vit import TransformerModel
from model.timm_vit import TimmVit
from model.deepnet import DeepNet

model_dict = {
    "resnet": CategoricalNetwork,
    "vit": TransformerModel,
    "timm_vit": TimmVit,
    "deepnet": DeepNet,
}