# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: This file contains some helper functions.
# Project: One/Few-Shot Learning for Galaxy Zoo

from logging import raiseExceptions
import os
from typing import NewType
import numpy as np
import pandas as pd

# I cannot import torch on the cluster using Jupyter Notebook
# We have to submit a script in order for it to find a GPU
# We installed PyTorch GPU
# So we are not importing it for normal tests (or we can install the cpu version for simple test?)
# No, we do not want to have both PyTorch GPU and PyTorch CPU
# PyTorch CPU is a subset of PyTorch GPU
# We have to uncomment parts of the code when submitting a job on the cluster

TorchTensor = NewType('TorchTensor', classmethod)


def tensor_to_dict(tensor: TorchTensor, keys: list) -> dict:
    """Convert a tensor to a dictionary given a list of keys

    Args:
        tensor (torch.tensor): The tensor to convert.
        keys (list): The list of keys.

    Returns:
        dict: A dictionary with the keys and values of the tensor.
    """
    return {key: tensor[i].item() for i, key in enumerate(keys)}


def dict_to_tensor(dictionary: dict, keys: list) -> TorchTensor:
    """Converts a dictionary to a tensor.

    Args:
        dictionary (dict): the dictionary to convert
        keys (list): the list of keys (usually in the setting file)

    Returns:
        torch.tensor: the pytorch tensor
    """

    return None  # torch.tensor([dictionary[key] for key in keys])


def subset_dict(dictionary: dict, keys: list) -> dict:
    """Generates a subset of a dictionary.

    Args:
        dictionary (dict): A long dictionary with keys and values respectively.
        keys (list): A list of keys to be extracted.

    Returns:
        dict: A dictionary with only the keys specified.
    """

    return {key: dictionary[key] for key in keys}


def store_arrays(array: np.ndarray, folder_name: str, file_name: str) -> None:
    """Stores a numpy array in a folder.

    Args:
        array (np.ndarray): The array to store.
        folder_name (str): The name of the folder.
        file_name (str): The name of the file.
    """

    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # use compressed format to store data
    np.savez_compressed(folder_name + '/' + file_name + '.npz', array)


def load_arrays(folder_name: str, file_name: str) -> np.ndarray:
    """Load the arrays from a folder.

    Args:
        folder_name (str): name of the folder.
        file_name (str): name of the file.

    Returns:
        np.ndarray: The array.
    """

    matrix = np.load(folder_name + '/' + file_name + '.npz')['arr_0']

    return matrix


def load_csv(folder_name: str, file_name: str) -> pd.DataFrame:
    """Given a folder name and file name, we will load the csv file.

    Args:
        folder_name(str): the name of the folder
        file_name(str): name of the file

    Returns:
        pd.DataFrame: the loaded csv file
    """
    path = folder_name + '/' + file_name + '.csv'

    if not os.path.isfile(path):
        raise FileNotFoundError('File not found: ' + path)

    else:
        df = pd.read_csv(path)
        return df


def save_csv(array: np.ndarray, folder_name: str, file_name: str) -> None:
    """Save an array to a csv file

    Args:
        array (np.ndarray): The array to be saved
        folder_name (str): The name of the folder
        file_name (str): The name of the file
    """
    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    np.savetxt(folder_name + '/' + file_name + '.csv', array, delimiter=',')


def save_pd_csv(df: pd.DataFrame, folder_name: str, file_name: str) -> None:
    """Save an array to a csv file

    Args:
        array (np.ndarray): The array to be saved
        folder_name (str): The name of the folder
        file_name (str): The name of the file
    """
    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    df.to_csv(folder_name + '/' + file_name + '.csv', index=False)


def save_parquet(df: pd.DataFrame, folder_name: str, file_name: str) -> None:
    """Save a dataframe to a parquet file

    Args:
        df(pd.DataFrame): The dataframe to be saved
        folder_name(str): The name of the folder
        file_name(str): The name of the file
    """
    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    df.to_parquet(folder_name + '/' + file_name + '.parquet', index=False)


def read_parquet(folder_name: str, file_name: str) -> pd.DataFrame:
    """Given a folder name and file name, we will load the parquet file.

    Args:
        folder_name(str): the name of the folder
        file_name(str): name of the file

    Returns:
        pd.DataFrame: the loaded csv file
    """
    path = folder_name + '/' + file_name + '.parquet'

    if not os.path.isfile(path):
        raise FileNotFoundError('File not found: ' + path)

    else:
        df = pd.read_parquet(path)
        return df
