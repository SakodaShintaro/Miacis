from model.resnet import CategoricalNetwork
from model.resnet import ScalarNetwork
from model.vit import TransformerModel
from model.timm_vit import TimmVit

model_dict = {
    "resnet": ScalarNetwork,
    "vit": TransformerModel,
    "timm_vit": TimmVit
}