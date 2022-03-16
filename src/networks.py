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
        self.backbone = models.__dict__[backbone](pretrained=False, progress=True)

        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Get the number of features that are outputted by the last layer of backbone network.
        out_features = list(self.backbone.modules())[-1].out_features

        # Create an MLP (multi-layer perceptron) as the classification head.
        # Classifies if provided combined feature vector of the 2 images represent the same image.
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

        self.cls_head = nn.Sequential(nn.Linear(out_features, 1), nn.Sigmoid(),)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
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
        combined_features = feat1 * feat2  # torch.abs(feat1 - feat2)  #

        # Pass the combined feature vector through classification head to get similarity value in the range of 0 to 1.
        output = self.cls_head(combined_features)

        return output
