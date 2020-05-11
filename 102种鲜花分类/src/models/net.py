# coding=utf-8
# author=yphacker


import torch.nn as nn
from torchvision.models import resnext50_32x4d, wide_resnet50_2, densenet121,densenet201
from efficientnet_pytorch import EfficientNet
from conf import config
from utils.inceptionv4 import inceptionv4
from utils.xception import xception
from utils.senet import se_resnet152


class Net(nn.Module):
    def __init__(self, model_name):
        super(Net, self).__init__()
        model = None
        if model_name == 'resnet':
            model = wide_resnet50_2(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, config.num_classes)
        if model_name == 'resnext':
            model = resnext50_32x4d(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, config.num_classes)
        elif model_name == 'inceptionv4':
            model = inceptionv4(pretrained='imagenet')
            model.last_linear = nn.Linear(model.last_linear.in_features, config.num_classes)
        elif model_name == 'xception':
            model = xception(pretrained='imagenet')
            model.last_linear = nn.Linear(model.last_linear.in_features, config.num_classes)
        elif model_name == 'densenet121':
            model = densenet201(pretrained=True)
            model.classifier = nn.Linear(1024, config.num_classes)
        elif model_name == 'senet':
            model = se_resnet152(num_classes=config.num_classes)
        elif model_name == 'efficientnet':
            model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=config.num_classes)

        self.model = model

    def forward(self, img):
        out = self.model(img)
        return out
