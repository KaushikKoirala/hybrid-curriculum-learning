import torch.nn as nn
from torchvision import models

def create_convNextT(config):

    if config["pretrained"]:
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    else:
        weights = None

    
    model = models.convnext_tiny(weights=weights)

    # Replace the classifier to output 100 classes instead of 10
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, config["num_classes"])

    return model