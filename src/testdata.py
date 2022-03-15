"""
Dataloader (test phase) for the Galaxy Zoo data using PyTorch.

# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Project: One/Few-Shot Learning for Galaxy Zoo
"""

import os
import glob
import time
from typing import Tuple

import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import IterableDataset
from PIL import Image

# our script and functions
import settings as st

class TestData(IterableDataset):

	def __init__(self, gz_path: str, test_path: str, shuffle: bool = False, augment: bool = False):

		# shuffle the pairs
		self.shuffle = shuffle

		# choose if we want to augment the data
		self.augment = augment

		# store the path
		self.gz_path = gz_path

		# transformations
		trans = st.transformation

		# if we choose to augment, we apply the horizontal flip
		if self.augment:

			trans.append(transforms.RandomHorizontalFlip(p=0.5))

		# create the transform
		self.transform = transforms.Compose(trans)

		# the locations of the galaxy images 
		self.gz_images = glob.glob(os.path.join(self.gz_path, "*/*.png"))[0:10]

		# number of images 
		self.nimages = len(self.gz_images)

		# the full path of the test image
		self.test_path = test_path

	def __iter__(self):

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







