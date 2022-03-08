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
        save (bool): save the file if we want to. Defaults to False.

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


def filtering(df: pd.DataFrame, dictionary: dict, save: bool = False, **kwargs) -> pd.DataFrame:
    """Given a dictionary of filters, filter the dataframe. For example,

    dictionary = {'has-spiral-arms_yes_fraction' : 0.75, 'has-spiral-arms_yes' : 20}

    means we have at least 20 volunteers, who have voted for spiral arms, and the fraction of those who voted for spiral arms is at least 0.75.

    Note that the keys in the dictionary are the column names in the dataframe.

    Args:
        df (pd.DataFrame): A pandas dataframe with the metadata
        dictionary (dict): A dictionary of filters.
        save (bool): Option to save the outputs. Defaults to False.

    Returns:
        pd.DataFrame: A pandas dataframe with the filtered data.
    """

    # number of objects in the dataframe
    nobjects = df.shape[0]

    # items in the dictionary
    items = list(dictionary.items())

    condition = [True] * nobjects

    for item in items:
        condition &= df[item[0]] > item[1]

    # apply condition and reset index
    df_sub = df[condition]
    df_sub.reset_index(inplace=True, drop=True)

    if save:
        filename = kwargs.pop('filename')
        hp.save_parquet(df_sub, st.data_dir + '/descriptions', filename)

    return df_sub


def subset_df(dataframe: pd.DataFrame, nsubjects: int, random: bool = False, save: bool = False, **kwargs) -> pd.DataFrame:
    """Generate a subset of objects, for example, 2000 out of 10 000 spirals.

    Args:
        dataframe (pd.DataFrame): A dataframe consisting of specific objects, for example, spirals.
        nsubjects (int): The number of subjects we want to pick.
        random (bool): We can set this to True, if we want to pick the subjects randomly.
        save (bool): Option to save the outputs. Defaults to False.

    Returns:
        pd.DataFrame: A pandas dataframe consisting of a subset of images.
    """

    # total number of objects
    total = dataframe.shape[0]

    assert nsubjects <= total, 'The number of subjects requested is larger than the available number of objects.'

    if random:
        idx = np.random.choice(total, nsubjects, replace=False)

    else:
        idx = range(nsubjects)

    df_sub = dataframe.iloc[idx]
    df_sub.reset_index(inplace=True, drop=True)

    if save:
        filename = kwargs.pop('filename')
        hp.save_parquet(df_sub, st.data_dir + '/descriptions', filename)

    return df_sub


def copy_images(df: pd.DataFrame, foldername: str) -> None:
    """Copy images from Mike's folder to our working directory.

    Args:
        df (pd.DataFrame): A dataframe consisting of specific objects, for example, spiral
        foldername (str): Name of the folder where we want to copy the images
    """

    # number of objects
    nobjects = df.shape[0]

    # create a folder where we want to store the images
    folder = st.data_dir + '/' + 'images' + '/' + foldername + '/'

    # create the different folders if they do not exist (remove them if they exist already)
    if os.path.exists(folder):

        # delete the folder first if it exists
        shutil.rmtree(folder)

    # then create a new one
    os.makedirs(folder)

    counts = 0
    # fetch the data from Mike's directory
    for i in range(nobjects):

        decals_file = st.decals + '/' + df['png_loc'].iloc[i]

        if os.path.isfile(decals_file):
            cmd = f'cp {decals_file} {folder}'
            os.system(cmd)
            counts += 1

    print(f'{counts} images saved to {folder}')


def split_data(tag_names: list, val_size: float = 0.20, test_size: float = 0.20, save: bool = False) -> dict:
    """Split the data into training and validation size for assessing the performance of the network.

    Args:
        tag_names (list): A list of the tag names, for example, elliptical, ring, spiral
        val_size (float, optional): The size of the validation set, a number between 0 and 1. Defaults to 0.20.
        test_size (float, optional): The size of the testing set, a number between 0 and 1. Defaults to 0.20.
        save (bool): Choose if we want to save the outputs generated. Defaults to False.

    Returns:
        dict: A dictionary consisting of the training and validation data.
    """
    
    # compute the training size 
    train_size = 1.0 - val_size - test_size 
    
    assert train_size > 0, "The validation and/or test size is too large."
    
    d = {}

    for item in tag_names:

        # load the csv file
        tag_file = hp.read_parquet(st.data_dir, 'descriptions/subset_' + item)

        # split the data into train and test        
        dummy, test = sm.train_test_split(tag_file, test_size=test_size)
        
        # the validation size needs to be updated based on the first split 
        val_new = val_size * tag_file.shape[0] / dummy.shape[0]
        
        # then we generate the training and validation set 
        train, validate = sm.train_test_split(dummy,test_size=val_new)

        # reset the index (not required, but just in case)
        test.reset_index(drop=True, inplace=True)
        train.reset_index(drop=True, inplace=True)
        validate.reset_index(drop=True, inplace=True)

        # store the dataframes in the dictionary
        d[item] = {'train': train, 'validate': validate, 'test': test}

        if save:
            hp.save_pd_csv(d[item]['train'], st.data_dir + '/' + 'ml/train', item)
            hp.save_pd_csv(d[item]['validate'], st.data_dir + '/' + 'ml/validate', item)
            hp.save_pd_csv(d[item]['test'], st.data_dir + '/' + 'ml/test', item)

    return d


def move_data(subset: str, object_type: str) -> None:
    """Move data to the right folder. 

    Args:
        subset (str) : validate or train or test 
        object_type (str): name of the object, for example, 'spiral', we want to move
    """
    
    assert subset in ['validate', 'test', 'train'], "Typical group in ML: validate, train, test"
    
    # the Machine Learning set (validate, train, test)
    ml_set = hp.load_csv(st.data_dir + '/ml/'+ subset, object_type)
       
    # number of objects we have 
    nobject = ml_set.shape[0]
    
    # folder where we want to store the images
    folder = st.data_dir + '/' + 'ml' + '/' + subset + '_images' + '/' + object_type + '/'
    
    # create the different folders if they do not exist (remove them if they exist already)
    if os.path.exists(folder):

        # delete the folder first if it exists
        shutil.rmtree(folder)

    # then create a new one
    os.makedirs(folder)
    
    # copy the data from images/item to categories/train/item
    for j in range(nobject):

        file = st.data_dir + '/' + 'images' + '/' + object_type + '/' + ml_set.iauname.iloc[j] + '.png'

        if os.path.isfile(file):
            cmd = f'cp {file} {folder}'
            os.system(cmd)

                
def images_train_validate_test(tag_names: list) -> None:
    """Read the csv file for a particular tag and copy the images in their respective folders

    Args:
        tag_names (list): A list of the tag names, for example, elliptical, ring, spiral
        save (bool, optional): Choose if we want to save the outputs generated. Defaults to False.
    """

    for item in tag_names:
        for subset in ['validate', 'train', 'test']:
            move_data(subset, item)


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
    tags_updated.reset_index(inplace=True, drop=True)

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
