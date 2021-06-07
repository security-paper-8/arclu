from collections import OrderedDict
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import torchvision.models as models


class MNISTModel(torch.nn.Module):
    def __init__(self, y_mu_encoding):
        super(MNISTModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 3, 1)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, 1)
        self.fc1 = torch.nn.Linear(5 * 5 * 128, 1024)
        self.fc2 = torch.nn.Linear(1024, y_mu_encoding.shape[1])
        self.y_mu_encoding = y_mu_encoding

    def forward(self, x):
        o = F.relu(self.conv1(x))
        o = F.max_pool2d(o, 2)
        o = F.relu(self.conv2(o))
        o = F.max_pool2d(o, 2)
        o = torch.flatten(o, 1)
        o = self.fc1(o)
        o = self.fc2(o)
        return o

    def compute_squared_distance(self, o):
        return torch.pow(o.unsqueeze(1) - self.y_mu_encoding.unsqueeze(0), 2).sum(dim=2)


class LightImageNetModel(torch.nn.Module):
    def __init__(self, y_mu_encoding, num_classes=20):
        super().__init__()
        self.num_latents = 1000
        self.num_classes = num_classes
        self.y_mu_encoding = y_mu_encoding

        # Load Resnet18
        self.model = models.resnet18(True)
        # Finetune Final few layers to adjust for light imagenet input
        self.model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.model.fc.out_features = 200

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

    def forward(self, x):
        return self.model(x)

    def compute_squared_distance(self, o):
        return torch.pow(o.unsqueeze(1) - self.y_mu_encoding.unsqueeze(0), 2).sum(dim=2)


class OursDataParallel(torch.nn.DataParallel):
    def compute_squared_distance(self, o):
        return torch.pow(o.unsqueeze(1) - self.module.y_mu_encoding.unsqueeze(0), 2).sum(dim=2)
