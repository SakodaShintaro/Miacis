from model.resnet import CategoricalNetwork
from model.vit import TransformerModel
from model.timm_vit import TimmVit

model_dict = {
    "resnet": CategoricalNetwork,
    "vit": TransformerModel,
    "timm_vit": TimmVit
}