# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: Given a test image, find similar objects.
# Project: One/Few-Shot Learning for Galaxy Zoo

import os
import time 
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# our scripts and functions
import utils.imaging as ui
import settings as st
import utils.helpers as hp
from src.testdata import TestData
from src.networks import SiameseNetwork

# we will normally evaluate on CPU (to maend if we want to predict on GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the model
loaded_model = torch.load('../fs-models/siamese_resnet18.pth')
model = SiameseNetwork(backbone="resnet18")
model.to(device)
model.load_state_dict(loaded_model)
model.eval()

decals = hp.load_pickle('supplement', 'image_locations')
test = '/home/phys2286/GAZFEW/test-images/J102532.37+052457.6.png'

test_dataset = TestData(decals, test)
test_dataloader = DataLoader(test_dataset, batch_size=32)

scores = list() 

t0 = time.time()

for i, (img1, img2) in enumerate(test_dataloader):

    img1, img2 = map(lambda x: x.to(device), [img1, img2])

    prob = model(img1, img2)

    # scores.append(prob.item())
    scores += prob.cpu().detach().view(-1).numpy().tolist()
    
    if (i+1)%100 == 0:
        print('Already tested {0} out of {1}'.format(i, test_dataset.nimages))

t1 = time.time()

print("Total time taken is {}".format(t1 - t0))

# a dataframe of the scores only
scores = pd.DataFrame(scores, columns=['scores'])
scores['iauname'] = list(map(lambda x: x.split(os.sep)[-1][:-4], test_dataset.gz_images))

scores.to_csv('test-images/test.csv')

