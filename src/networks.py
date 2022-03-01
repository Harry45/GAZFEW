# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: Siamese network architectures for training.
# Project: One/Few-Shot Learning for Galaxy Zoo

import torch
import torch.nn as nn
from torchvision import models


class SiameseNetwork(nn.Module):
    def __init__(self, backbone="resnet18"):
        super(SiameseNetwork, self).__init__()

        if backbone not in models.__dict__:
            raise Exception("No model named {} exists in torchvision.models.".format(backbone))

        # Create a backbone network from the pretrained models provided in torchvision.models
        self.backbone = models.__dict__[backbone](pretrained=True, progress=True)

        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Get the number of features that are outputted by the last layer of backbone network.
        out_features = list(self.backbone.modules())[-1].out_features

        # Create an MLP (multi-layer perceptron) as the classification head.
        # Classifies if provided combined feature vector of the 2 images represent same player or different.
        # self.cls_head = nn.Sequential(
        #     nn.Dropout(p=0.5),
        #     nn.Linear(out_features, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),

        #     nn.Dropout(p=0.5),
        #     nn.Linear(512, 64),
        #     nn.BatchNorm1d(64),
        #     nn.Sigmoid(),
        #     nn.Dropout(p=0.5),

        #     nn.Linear(64, 1),
        #     nn.Sigmoid(),
        # )

        self.cls_head = nn.Sequential(nn.Linear(out_features, 512), nn.Sigmoid(),)

    def forward(self, img1: torch.tensor, img2: torch.tensor) -> torch.tensor:
        """Calculates the similarity between two images.

        Args:
            img1 (torch.tensor): The first image
            img2 (torch.tensor): The second image

        Returns:
            torch.tensor: The similarity between the two images between 0 and 1
        """
        # Pass the both images through the backbone network to get their separate feature vectors
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)

        # Multiply (element-wise) the feature vectors of the two images together,
        # to generate a combined feature vector representing the similarity between the two.
        combined_features = feat1 * feat2

        # Pass the combined feature vector through classification head to get similarity value in the range of 0 to 1.
        output = self.cls_head(combined_features)

        return output


class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 10),  # 64@96*96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64@48*48
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),    # 128@42*42
            nn.MaxPool2d(2),   # 128@21*21
            nn.Conv2d(128, 128, 4),
            nn.ReLU(),  # 128@18*18
            nn.MaxPool2d(2),  # 128@9*9
            nn.Conv2d(128, 256, 4),
            nn.ReLU(),   # 256@6*6
        )
        self.linear = nn.Sequential(nn.Linear(102400, 4096), nn.Sigmoid())
        self.out = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        return self.sigmoid(out)
        # return out
