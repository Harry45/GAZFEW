# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: Given a test image, find similar objects.
# Project: One/Few-Shot Learning for Galaxy Zoo

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# our scripts and functions
import utils.imaging as ui
import settings as st
import utils.helpers as hp
from src.dataset import DataSet
from src.networks import SiameseNetwork

# we will normally evaluate on CPU (to maend if we want to predict on GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the model
loaded_model = torch.load('../fs-models/siamese_resnet18.pth')
model = SiameseNetwork(backbone="resnet18")
model.to(device)
model.load_state_dict(loaded_model)
model.eval()

test_dataset = DataSet(st.test_path, shuffle=False, augment=False)
test_dataloader = DataLoader(test_dataset, batch_size=1)

criterion = torch.nn.BCELoss()

losses = []
correct = 0
total = 0

for i, ((img1, img2), y, (class1, class2)) in enumerate(test_dataloader):

    print("[{} / {}]".format(i, len(test_dataloader)))

    img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])
    
    class1 = class1[0]
    class2 = class2[0]

    prob = model(img1, img2)
    loss = criterion(prob, y)

    losses.append(loss.item())
    correct += torch.count_nonzero(y == (prob > 0.5)).item()
    total += len(y)
    
print("Test: Loss={:.2f}\t Accuracy={:.2f}\t".format(sum(losses)/len(losses), correct / total))

