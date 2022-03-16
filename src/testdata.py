"""
Dataloader (test phase) for the Galaxy Zoo data using PyTorch.

# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Project: One/Few-Shot Learning for Galaxy Zoo
"""

from typing import Tuple
import torch
from torchvision import transforms
from torch.utils.data import IterableDataset
from PIL import Image

# our script and functions
import settings as st


class TestData(IterableDataset):
    """Data loader for the test images.

    Args:
        gz_path (str): the path to the full galaxy zoo dataset
        test_path (str): the location of the test image
    """

    def __init__(self, gz_path: str, test_path: str):

        # store the path
        self.gz_path = gz_path

        # transformations
        trans = st.transformation

        # create the transform
        self.transform = transforms.Compose(trans)

        # the locations of the galaxy images
        # self.gz_images = glob.glob(os.path.join(self.gz_path, "*/*.png"))[0:500]
        self.gz_images = gz_path[0:500]

        # number of images
        self.nimages = len(self.gz_images)

        # the full path of the test image
        self.test_path = test_path

    def __iter__(self):
        """Generates a pair of images.

        Yields:
            Two images to be compared
        """

        for idx in range(self.nimages):

            # get the image paths for the pair
            image_path1 = self.test_path
            image_path2 = self.gz_images[idx]

            # load the two images
            image1 = Image.open(image_path1).convert("RGB")
            image2 = Image.open(image_path2).convert("RGB")

            # transform the images
            if self.transform:
                image1 = self.transform(image1).float()
                image2 = self.transform(image2).float()

            yield (image1, image2)

    def __len__(self):
        """Return the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.gz_images)
