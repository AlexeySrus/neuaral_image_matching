import torch
import torch.nn as nn
from torch.nn import *
from image_matching.architectures.resnet import resnet101


class MatchModel(nn.Module):
    def __init__(self):
        super(MatchModel, self).__init__()

        self.resnet = resnet101(True)

        self.layer1 = nn.Linear(2048*2, 64)
        self.layer2 = nn.Linear(64, 9)

    def forward(self, original, transformed):
        original_features = self.resnet(original)
        transformed_features = self.resnet(transformed)

        original_x = original_features.view(-1, 2048)
        transformed_x = transformed_features.view(-1, 2048)

        x = torch.cat((original_x, transformed_x), dim=1)

        x = self.layer1(x)
        x = nn.ReLU()(x)
        x = self.layer2(x)
        x = x.view(-1, 3, 3)

        return x
