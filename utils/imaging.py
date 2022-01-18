# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: Some functions to process the data, for example, which will help us make selections
# Project: One/Few-Shot Learning for Galaxy Zoo

from typing import Tuple, NewType
import PIL 
import numpy as np

PILimage = NewType('PILimage', classmethod)

def object_name(iauname: str) -> Tuple[str, str]:
    """The data in Mike's folder is organised by folder name and image name. The format is such that
    the first few letters correspond to the folder name and the the rest as the image name.

    Args:
        iauname (str): name of image in the csv file containing the tags

    Returns:
        Tuple[str, str]: folder name and file name
    """

    # the folder name is just the first four letters in the image name
    folder = iauname[0:4]

    return folder, iauname + '.png'


def load_image(data_dir: str, filename: str) -> Tuple[PILimage, np.ndarray]:
    """Load the image in both PIL and numpy format. We might want to use the numpy array only.

    Args:
        data_dir (str): the full path where the image is located
        filename (str): the name of the image

    Returns:
        Tuple[PIL.PngImagePlugin.PngImageFile, np.ndarray]: PIL image with all descriptions, numpy array of the image
    """

    # image in PIL format
    im_pil = PIL.Image.open(data_dir + filename)

    # image as a numpy array
    im_arr = np.asarray(im_pil)

    return im_pil, im_arr
