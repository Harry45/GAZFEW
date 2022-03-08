# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: Given a test image, find similar objects.
# Project: One/Few-Shot Learning for Galaxy Zoo
# NOTE: this script is still work in progress!

import pandas as pd
import torch
from torchvision import transforms

# our scripts and functions
import utils.imaging as ui
import settings as st
import utils.helpers as hp
from src.networks import SiameseNetwork


def load_ml_model(path: str) -> classmethod:
    """Load the model from the path.

    Args:
        path (str): the path to the model

    Returns:
        classmethod: the model
    """

    # we will normally evaluate on CPU (to maend if we want to predict on GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # the Siamese network we are using
    model = SiameseNetwork(backbone="resnet18")

    # load the model
    loaded_model = torch.load(path, map_location=device)
    model.load_state_dict(loaded_model)

    # evaluate the load
    model.eval()

    return model, device


def calculate_scores(
        test_image: str, dataframe: pd.DataFrame, model: classmethod, device: str, save: bool = False) -> pd.DataFrame:
    """Given an image, we will iterate through each object in the dataframe and calculate a similarity score.
    The dataframe is expected to have columns consisting of the columns:
    - 'iauname'
    - 'ra'
    - 'dec'
    - 'redshift'
    - 'png_loc'

    Args:
        test_image (str): the test image, assuming it is in the test-images/ folder
        dataframe (pd.dataFrame): a dataframe consisting of all the other images
        model (classmethod): the deep learning model to use for the similarity score
        save (bool, optional): Option to save the generated scores. Defaults to False.

    Returns:
        pd.DataFrame: A dataframe with the scores for each image.
    """

    # get the test image
    if '.png' not in test_image:
        test_image = test_image + '.png'

    # load the image using our function
    pil, _ = ui.load_image_full('test-images/' + test_image)

    # create the transformation class
    transformation = transforms.Compose(st.transformation)

    # apply transformation to the first image
    img1_trans = transformation(pil).float()

    scores = list()

    ntest = 100

    for image in dataframe.png_loc.values[0:ntest]:

        # load the second image
        pil2, _ = ui.load_image_full(st.decals + '/' + image)

        img2_trans = transformation(pil2).float()

        scores.append(model(img1_trans.unsqueeze(1), img2_trans.unsqueeze(1)).item())

    # a dataframe of the scores only
    scores = pd.DataFrame(scores, columns=['scores'])

    # the meta data for the objects
    metadata = dataframe.iloc[0:ntest][['iauname', 'ra', 'dec', 'redshift', 'png_loc']]

    # the final dataframe
    results = pd.concat([metadata, scores], axis=1)

    # create a dictionary with the key being the object name and the value being the score
    dictionary = {test_image[0:-4]: results}

    if save:
        hp.save_pickle(dictionary, 'test-images', test_image[0:-4])

    return dictionary
