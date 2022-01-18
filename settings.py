# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: This file contain the main settings for running the codes.
# Project: One/Few-Shot Learning for Galaxy Zoo

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

# We will also work with spirals, rings and ellipticals.
