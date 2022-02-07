# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: Dataloader for PyTorch.
# Project: One/Few-Shot Learning for Galaxy Zoo

import os
import glob
import time

import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import IterableDataset
from PIL import Image

# our script and functions
import settings as st


class DataSet(IterableDataset):

    def __init__(self, path: str, shuffle: bool = False, augment: bool = False, normalise: bool = False):

        # shuffle the pairs
        self.shuffle = shuffle

        # choose if we want to augment the data
        self.augment = augment

        # store the path
        self.path = path

        # store the choice of normalisation
        self.normalise = normalise

        # this is the default transform
        # trans = [transforms.ToTensor(), transforms.Resize(st.new_img_size[1:])]
        trans = [transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
                 transforms.Resize(300), transforms.CenterCrop(224)]

        # apply normalisation if set in argument
        if self.normalise:
            trans.append(transforms.Normalize(mean=st.mean_img, std=st.std_img))

        # if we choose to augment, we apply the horizontal flip
        if self.augment:

            trans.append(transforms.RandomHorizontalFlip(p=0.5))

        # create the transform
        self.transform = transforms.Compose(trans)

        # create the pairs
        self.create_pairs()

    def create_pairs(self):
        """Create the pairs of images and classes."""

        # gather all the images in the folder
        self.image_paths = glob.glob(os.path.join(self.path, "*/*.png"))

        # empty list to create the image class
        self.image_classes = []

        # a dictionary to store the class indices
        self.class_indices = {}

        for image_path in self.image_paths:

            # we get the class of the image, for example, elliptical, ring, spiral
            image_class = image_path.split(os.path.sep)[-2]

            # we add the class to the list
            self.image_classes.append(image_class)

            if image_class not in self.class_indices:
                self.class_indices[image_class] = []

            # we add the index of the image to the class
            self.class_indices[image_class].append(self.image_paths.index(image_path))

        # create a set of indices
        self.indices1 = np.arange(len(self.image_paths))

        if self.shuffle:
            np.random.seed(int(time.time()))
            np.random.shuffle(self.indices1)
        else:
            # If shuffling is set to off, set the random seed to 1, to make it deterministic.
            np.random.seed(1)

        # generate a set of boolean values, for which all numbers below the threshold are true, and all above are false
        select_pos_pair = np.random.rand(len(self.image_paths)) < 0.5

        # create the indices
        self.indices2 = []

        for i, pos in zip(self.indices1, select_pos_pair):

            # this is the index of the image for class 1
            class1 = self.image_classes[i]

            # if positive, class 2 is the same as class 1
            # otherwise, class 2 is different from class 1
            if pos:
                class2 = class1
            else:
                class2 = np.random.choice(list(set(self.class_indices.keys()) - {class1}))

            # we get the index of the image for class 2
            idx2 = np.random.choice(self.class_indices[class2])

            # we add the index to the list
            self.indices2.append(idx2)

        # convert to numpy array
        self.indices2 = np.array(self.indices2)

    def __iter__(self):

        for idx, idx2 in zip(self.indices1, self.indices2):

            # get the image paths for the pair
            image_path1 = self.image_paths[idx]
            image_path2 = self.image_paths[idx2]

            # get the classes for the pair
            class1 = self.image_classes[idx]
            class2 = self.image_classes[idx2]

            # load the two images
            image1 = Image.open(image_path1).convert("RGB")
            image2 = Image.open(image_path2).convert("RGB")

            # transform the images
            if self.transform:
                image1 = self.transform(image1).float()
                image2 = self.transform(image2).float()

            yield (image1, image2), torch.FloatTensor([class1 == class2]), (class1, class2)

    def __len__(self):
        return len(self.image_paths)
