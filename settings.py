# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: This file contain the main settings for running the codes.
# Project: One/Few-Shot Learning for Galaxy Zoo

# Steps
# 1) Process the tag file which Mike shared
# 2) Find the set of images which exists in the DECaLS DR5 folder, using the csv file containing the tags.
# 3) Process the meta data so we can extract RA, DEC, NSA_ID, and REDSHIFT
# 4) Output a csv file with the subset of images which exists in the DECaLS DR5 folder ($DATA/data/tags_images.csv).
# 5) Group the images by tag names and output csv files with the images grouped by tag names ($DATA/data/tags_spiral.csv).
# 6) Copy the data (per category) from Mike's folder to the new folder ($DATA/data/category/spiral/object.jpg).

# DECaLS (the data is in Mike's directory on ARC cluster)
decals = '/data/phys-zooniverse/chri5177/galaxy_zoo/decals/dr5/png'

# Data from my folder (will also contains the tags that Mike shared)
data_dir = '/data/phys-zooniverse/phys2286/data'

# column names to keep
# the tags seem to have been generated from different data sources
keys_1 = ["!ra", "!dec", "!nsa_id", "!Z"]
keys_2 = ["!ra", "!dec", "!nsa_id", "!redshift"]
consistent_keys = ['ra', 'dec', 'nsa_id', 'redshift']
colnames = ['tag', 'user_id', 'subject_id', 'image_url', 'iauname', 'exists'] + consistent_keys
