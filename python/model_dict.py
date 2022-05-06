from model.resnet import CategoricalNetwork
from model.vit import TransformerModel

model_dict = {
    "resnet": CategoricalNetwork,
    "vit": TransformerModel
}