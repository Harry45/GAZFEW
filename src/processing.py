# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: This file is for processing the data (tags) and making selections.
# Project: One/Few-Shot Learning for Galaxy Zoo

import os
import json
import pandas as pd

# our own functions and scripts
import utils.imaging as ui
import utils.helpers as hp
import settings as st


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
        hp.save_pd_csv(tags_updated, st.data_dir, filename)

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
        hp.save_pd_csv(df_cat, st.data_dir, filename)

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
    assert 'tag' in list(df.colnames), 'The dataframe should contain a column with name "tag".'

    dictionary = {}
    for item in tag_names:
        dictionary[item] = df[df['tag'] == item]

        if save:
            hp.save_pd_csv(dictionary[item], st.data_dir, 'tags_images_' + item)

    return dictionary
