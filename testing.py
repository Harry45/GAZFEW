# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: Given a test image, find similar objects.
# Project: One/Few-Shot Learning for Galaxy Zoo

import os
import glob
import time
import pandas as pd
import torch
from torch.utils.data import DataLoader

# our scripts and functions
import utils.helpers as hp
from src.testdata import TestData
from src.networks import SiameseNetwork


def calculate_scores(folder: str, bs: int = 64):
    """Use the deep learning model to calculate the similarity scores.

    Args:
        folder (str): path to the folder
        bs (int): the batch size. Default: 64
    """

    path = os.path.join(os.getcwd(), folder, '*.png')
    imgs = glob.glob(path)

    # choice of device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the model
    loaded_model = torch.load('../fs-models/siamese_resnet18.pth')
    model = SiameseNetwork(backbone="resnet18")
    model.to(device)
    model.load_state_dict(loaded_model)
    model.eval()

    decals = hp.load_pickle('supplement', 'image_locations')

    for img in imgs:

        test_dataset = TestData(decals, img)
        test_dataloader = DataLoader(test_dataset, batch_size=bs)

        scores = list()

        start = time.time()

        for i, (img1, img2) in enumerate(test_dataloader):

            img1, img2 = map(lambda x: x.to(device), [img1, img2])

            prob = model(img1, img2)

            scores += prob.cpu().detach().view(-1).numpy().tolist()

            if (i + 1) % 100 == 0:
                print('Already tested {0} out of {1}'.format(i + 1, test_dataset.nimages))

        final = time.time()

        print("Total time taken is {}".format(final - start))

        # a dataframe of the scores only
        dataframe = pd.DataFrame()
        dataframe['iauname'] = list(map(lambda x: x.split(os.sep)[-1][:-4], test_dataset.gz_images))
        dataframe['scores'] = scores

        # save the scores
        img_name = os.path.split(img)[-1][:-4]
        hp.save_pickle(dataframe, folder, img_name + '.scores')


if __name__ == '__main__':
    calculate_scores('test-images', bs=64)
