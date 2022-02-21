# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: This file is for processing the data (tags) and making selections.
# Project: One/Few-Shot Learning for Galaxy Zoo

import os
import json
import shutil
import pandas as pd
import numpy as np
import sklearn.model_selection as sm

# our own functions and scripts
import utils.imaging as ui
import utils.helpers as hp
import settings as st


def correct_location(csv: str, save: bool = False, **kwargs) -> pd.DataFrame:
    """Rename the columns containing the image location to the right one.

    Args:
        csv (str): the name of the csv file
        save (bool): save the file if we want to

    Returns:
        pd.DataFrame: a dataframe consisting of the corrected location.
    """
    df = hp.load_csv(st.zenodo, csv)

    # the number of objects
    nobjects = df.shape[0]

    # The png images are in the folder png/Jxxx/ rather than dr5/Jxxx/
    locations = [df.png_loc.values[i][4:] for i in range(nobjects)]
    df.png_loc = locations

    # check if all files exist
    imgs_exists = [int(os.path.isfile(st.decals + '/' + locations[i])) for i in range(nobjects)]
    imgs_exists = pd.DataFrame(imgs_exists, columns=['exists'])
    df = pd.concat([df, imgs_exists], axis=1)

    if save:
        filename = kwargs.pop('filename')
        hp.save_parquet(df, st.data_dir + '/descriptions', filename)

    return df

# The code below was written when we were using the tags to make the selection of the images.


def find_exist_img(tag_file: str, save: bool = False, **kwargs) -> pd.DataFrame:
    """Find the set of images which exists in the DECaLS DR5 folder, using the csv file containing the tags.

    Args:
        tag_file (str): name of the tag file.

    Returns:
        pd.DataFrame: a dataframe with a reduced set of rows which exists in DR5.
    """

    assert 'iauname' in tag_file.columns, 'Column names iauname not in csv file!'

    record_exists = []

    for i in range(tag_file.shape[0]):

        # get the folder name and filename given an IAUname
        folder, fname = ui.object_name(tag_file.iauname.iloc[i])

        state = os.path.isfile(st.decals + '/' + folder + '/' + fname)

        if state:
            record_exists.append(1)
        else:
            record_exists.append(0)

    # convert list to dataframe
    exists_df = pd.DataFrame(record_exists, columns=['exists'])

    # concatenate two dataframes
    tags_new = pd.concat([tag_file, exists_df], axis=1)

    # subset (the ones for which we have an image)
    tags_updated = tags_new[tags_new['exists'] == 1]
    tags_updated.reset_index(inplace=True)

    if save:
        filename = kwargs.pop('filename')
        hp.save_pd_csv(tags_updated, st.data_dir + '/tags', filename)

    return tags_updated


def process_meta(df: pd.DataFrame, save: bool = False, **kwargs) -> pd.DataFrame:
    """The metadata in the csv file contain further information on the subject.
    We just process the file in a more representable format, keeping RA, DEC, REDSHIFT and NSA_ID

    Args:
        df (pd.DataFrame): A pandas dataframe with the metadata

    Returns:
        pd.DataFrame: A pandas dataframe with the relevant keys only. See settings file.
    """
    def func(x): return json.loads(x.replace("'", "\""))

    record_meta = []

    for i in range(df.shape[0]):
        info = func(df.metadata.iloc[i])
        try:
            d = [info[key] for key in st.keys_1]
        except:
            d = [info[key] for key in st.keys_2]

        record_meta.append(d)

    # convert list to dataframe and rename the columns
    meta_df = pd.DataFrame(record_meta, columns=['ra', 'dec', 'nsa_id', 'redshift'])

    # concatenate the original file and metadata and we keep certain columns only
    df_cat = pd.concat([df, meta_df], axis=1)[st.colnames]

    if save:
        filename = kwargs.pop('filename')
        hp.save_pd_csv(df_cat, st.data_dir + '/tags', filename)

    return df_cat


def select_df_tags(df: pd.DataFrame, tag_names: list, save: bool = False) -> dict:
    """Given a dataframe which contains the details of the different images and a list
    of the different tags, we will create subsets of the main dataframe and choose to store them.
    There are more than 3000 tags but here we will focus on the most popular ones, for example,

    - spiral
    - elliptical
    - ring
    - bar

    which are common in galaxy morphological classifications.

    Args:
        df (pd.DataFrame): The main dataframe with the details of all images.
        tag_names (list): A list of tag names.
        save (bool, optional): Choice of saving the outputs to a folder. See settings file. Defaults to False.
    Returns:
        dict: A dictionary consisting of the different dataframes.
    """
    assert len(tag_names) >= 1, 'At least one tag should be provided in a list, for example, ["spiral"].'
    assert 'tag' in list(df.columns), 'The dataframe should contain a column with name "tag".'

    dictionary = {}
    for item in tag_names:

        # select rows with the tag
        df_selected = df[df['tag'] == item]

        # there are also duplicate rows in the file - we keep the first one
        dictionary[item] = df_selected.drop_duplicates(subset=['iauname'], keep='first')

        if save:
            hp.save_pd_csv(dictionary[item], st.data_dir + '/tags', 'tags_images_' + item)

    return dictionary


def search_save_database(tag_name: str) -> None:
    """Search the database for the images which have the tag name and save them to a folder.

    Args:
        tag_name (str): The name of the tag.
    """

    # load the csv file from the tags/ folder
    df = hp.load_csv(st.data_dir + '/' + 'tags', 'tags_images_' + tag_name)

    # number of objects
    nobjects = df.shape[0]

    # create a folder where we want to store the images
    mainfolder = st.data_dir + '/' + 'categories' + '/' + tag_name + '/'

    if not os.path.exists(mainfolder):
        os.makedirs(mainfolder)

    counts = 0
    # fetch the data from Mike's directory
    for i in range(nobjects):
        folder, fname = ui.object_name(df.iauname.iloc[i])

        decals_file = st.decals + '/' + folder + '/' + fname

        if os.path.isfile(decals_file):
            cmd = f'cp {decals_file} {mainfolder}'
            os.system(cmd)
            counts += 1

    print(f'{counts} images saved to {mainfolder}')


def generate_random_set(tag_names: list, n_examples: int, save: bool = False) -> None:
    """Given a list of tags, we will generate a random set of n examples and store the images in
    categories/ and a csv file will also be generated in the tags/ folder. Note that, we are assuming
    that the images are already in the categories/ folder. The csv files per tag are also assumed to be in
    the tags/ folder. Note that

    - there are not repetitions,
    - the number of examples should be less or equal to the minimum number of images per tag.

    Args:
        tag_names (list): a list of the different tags, for example, ['spiral', 'ring', 'elliptical'].
        n_examples (int): the number of examples we want to use.
        save (bool): Choose if we want to save the outputs generated. Defaults to False.
    """

    # find the number of images per tag
    nobjects = [hp.load_csv(st.data_dir + '/tags', 'tags_images_' + item).shape[0] for item in tag_names]

    print(f'The number of examples we have is {nobjects}')

    assert n_examples <= min(nobjects), 'The number of examples should be less than the number of objects.'

    for i, item in enumerate(tag_names):

        # generate a set of unique random index numbers
        idx = np.random.choice(nobjects[i], n_examples, replace=False)

        # load the csv file
        df = hp.load_csv(st.data_dir + '/tags', 'tags_images_' + item).iloc[idx]

        # create a folder where we want to store the images
        mainfolder = st.data_dir + '/' + 'categories' + '/' + 'subset_' + item + '/'

        if os.path.exists(mainfolder):

            # delete the folder first if it exists
            shutil.rmtree(mainfolder)

        # then create a new one
        os.makedirs(mainfolder)

        counts = 0
        # fetch the data from Mike's directory
        for j in range(n_examples):

            folder, fname = ui.object_name(df.iauname.iloc[j])

            decals_file = st.decals + '/' + folder + '/' + fname

            if os.path.isfile(decals_file):
                cmd = f'cp {decals_file} {mainfolder}'
                os.system(cmd)
                counts += 1

        print(f'{counts} images saved to {mainfolder}')

        # save the dataframe to the folder tags/
        if save:
            hp.save_pd_csv(df, st.data_dir + '/tags', 'tags_images_subset_' + item)


def split_data(tag_names: list, val_size: float = 0.35, save: bool = False) -> dict:
    """Split the data into training and validation size for assessing the performance of the network.

    Args:
        tag_names (list): A list of the tag names, for example, elliptical, ring, spiral
        val_size (float, optional): The size of the validation set, a number between 0 and 1. Defaults to 0.35.
        save (bool): Choose if we want to save the outputs generated. Defaults to False.

    Returns:
        dict: A dictionary consisting of the training and validation data.
    """

    d = {}

    for item in tag_names:

        # load the csv file
        tag_file = hp.load_csv(st.data_dir + '/tags', 'tags_images_subset_' + item)

        # split the data into train and validation
        train, validate = sm.train_test_split(tag_file, test_size=val_size)

        # reset the index (not required, but just in case)
        train.reset_index(drop=True, inplace=True)
        validate.reset_index(drop=True, inplace=True)

        # store the dataframes in the dictionary
        d[item] = {'train': train, 'validate': validate}

        if save:
            hp.save_pd_csv(d[item]['train'], st.data_dir + '/ml/train_tags', item)
            hp.save_pd_csv(d[item]['validate'], st.data_dir + '/ml/validate_tags', item)

    return d


def images_train_validate(tag_names: list) -> None:
    """Read the csv file for a particular tag and copy the images in their respective folders

    Args:
        tag_names (list): A list of the tag names, for example, elliptical, ring, spiral
        save (bool, optional): Choose if we want to save the outputs generated. Defaults to False.
    """

    for item in tag_names:

        # read the training and validation file for a particular tag
        training = hp.load_csv(st.data_dir + '/ml/train_tags', item)
        validation = hp.load_csv(st.data_dir + '/ml/validate_tags', item)

        # number of items in training and validation
        ntrain = training.shape[0]
        nvalidate = validation.shape[0]

        # create a folder where we want to store the images
        train_folder = st.data_dir + '/' + 'ml' + '/' + 'train_images' + '/' + item + '/'
        val_folder = st.data_dir + '/' + 'ml' + '/' + 'validate_images' + '/' + item + '/'

        # create the different folders if they do not exist (remove them if they exist already)
        if os.path.exists(train_folder):

            # delete the folder first if it exists
            shutil.rmtree(train_folder)

        # then create a new one
        os.makedirs(train_folder)

        if os.path.exists(val_folder):

            # delete the folder first if it exists
            shutil.rmtree(val_folder)

        # then create a new one
        os.makedirs(val_folder)

        # copy the data from categories/subset_item to categories/train/item
        for j in range(ntrain):

            file = st.data_dir + '/' + 'categories' + '/' + 'subset_' + item + '/' + training.iauname.iloc[j] + '.png'

            if os.path.isfile(file):
                cmd = f'cp {file} {train_folder}'
                os.system(cmd)

        # copy the data from categories/subset_item to categories/validate/item
        for j in range(nvalidate):

            file = st.data_dir + '/' + 'categories' + '/' + 'subset_' + item + '/' + validation.iauname.iloc[j] + '.png'

            if os.path.isfile(file):
                cmd = f'cp {file} {val_folder}'
                os.system(cmd)
