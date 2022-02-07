# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: This file contain the main settings for running the codes.
# Project: One/Few-Shot Learning for Galaxy Zoo

# Steps
# 1) Process the tag file which Mike shared.
# 2) Find the set of images which exists in the DECaLS DR5 folder, using the csv file containing the tags.
# 3) Process the meta data so we can extract RA, DEC, NSA_ID, and REDSHIFT.
# 4) Output a csv file with the subset of images which exists in the DECaLS DR5 folder ($DATA/data/tags_images.csv).
# 5) Group the images by tag names and output csv files with the images grouped by tag names ($DATA/data/tags_spiral.csv).
# 6) Copy the data (per category) from Mike's folder to the new folder ($DATA/data/categories/spiral/object.jpg).
# 7) Draw N random samples of the images per tag and output the images to the new folder ($DATA/data/categories/subset_spiral/object_sample.jpg).

# Information
# ---------------------------------------------------------------------
#              | examples | size   | location
# ---------------------------------------------------------------------
# Spiral       | 6830     | 2.1 GB | $DATA/data/categories/spiral
# Ring         | 2735     | 0.8 GB | $DATA/data/categories/ring
# Elliptical   | 943      | 0.3 GB | $DATA/data/categories/elliptical
# ---------------------------------------------------------------------

# DECaLS (the data is in Mike's directory on ARC cluster)
decals = '/data/phys-zooniverse/chri5177/galaxy_zoo/decals/dr5/png'

# Data from my folder (will also contains the tags that Mike shared)
data_dir = '/data/phys-zooniverse/phys2286/data'
# data_dir = '/home/harry/Documents/Oxford/Astrophysics/Projects/Deep-Learning/data'

# column names to keep
# the tags seem to have been generated from different data sources
keys_1 = ["!ra", "!dec", "!nsa_id", "!Z"]
keys_2 = ["!ra", "!dec", "!nsa_id", "!redshift"]
consistent_keys = ['ra', 'dec', 'nsa_id', 'redshift']
colnames = ['tag', 'user_id', 'subject_id', 'image_url', 'iauname', 'exists'] + consistent_keys

# ---------------------------------------------------------------------
# the Deep Learning part
new_img_size = [3, 224, 224]

# basic statistics of the images. These are fixed, meaning same transformation should be applied to
# training, validation and test data.

# mean of the whole dataset (this is for 3 channels)
mean_img = [0.485, 0.456, 0.406]  # [26.97003201762193, 25.172733883647798, 24.687282796368816]

# standard deviation of the whole dataset
std_img = [0.229, 0.224, 0.225]  # [27.974221728738513, 25.714420641820155, 24.653711141402653]

# training and validation paths
train_path = data_dir + '/ml/train_images/'
val_path = data_dir + '/ml/validate_images/'
