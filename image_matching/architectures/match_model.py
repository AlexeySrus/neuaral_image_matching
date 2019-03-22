import torch
import torch.nn as nn
from torch.nn import *
from image_matching.architectures.resnet import resnet18


class MatchModel(nn.Module):
    def __init__(self):
        super(MatchModel, self).__init__()

        self.resnet = resnet18()
        self.layer1 = nn.Linear(1024, 9)

    def forward(self, original, transformed):
        original_features = self.resnet(original)
        transformed_features = self.resnet(transformed)

        print(original.shape)

        original_x = original_features.view(-1, 1024)
        transformed_x = transformed_features.view(-1, 1024)

        x = torch.cat((original_x, transformed_x), dim=1)

        x = self.layer1(x)
        x = x.view(-1, 3, 3)

        return x
