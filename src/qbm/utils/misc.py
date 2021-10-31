import os
import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from numpy.random import MT19937, RandomState, SeedSequence


def compute_df_stats(df):
    """
    Compute the min, max, mean, median, and standard deviation of the columns in the dataframe.

    :param df: Dataframe.

    :returns: Dataframe of the statistics.
    """
    return pd.DataFrame(
        {
            "min": df.min(),
            "max": df.max(),
            "mean": df.mean(),
            "median": df.median(),
            "std": df.std(),
        }
    ).T


def compute_stats_over_dfs(dfs):
    """
    Computes the means, medians, and standard deviations column/row-wise over the input
    list of dataframes.

    :param dfs: List of dataframes with identical row/column names.

    :returns: Dictionary of dataframes with the means, medians, and standard deviations.
    """
    df = pd.concat(dfs)
    means = df.groupby(df.index).mean()
    medians = df.groupby(df.index).median()
    stds = df.groupby(df.index).std()

    return {"means": means, "medians": medians, "stds": stds}


def save_artifact(artifact, file_path):
    """
    Saves a pickled artifact.

    :param artifact: Python object to save.
    :param file_path: Name of the file to save.
    """
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)

    with open(file_path, "wb") as f:
        pickle.dump(artifact, f)


def load_artifact(file_path):
    """
    Loads a pickled artifact.

    :param file_path: Name of the file to load.

    :returns: Loaded python object.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


@np.vectorize
def lr_exp_decay(epoch, decay_epoch, period):
    """
    Exponential decay function for use in learning rate scheduling.

    :param epoch: Current epoch.
    :param decay_epoch: Epoch at which to begin the decay.
    :param period: Decay period.

    :returns: The learning rate scaling factor.
    """
    return 2 ** (min((decay_epoch - epoch), 0) / period)


def get_rng(seed):
    """
    Creates a random number generator with the specified seed value.

    :param seed: Seed value for the rng.

    :returns: Numpy RandomState object.
    """
    return RandomState(MT19937(SeedSequence(seed)))


def get_project_dir():
    """
    Gets the project directory path from the environment and checks if it is valid.

    :returns: Path.
    """
    dir_path = os.getenv("QBM_PROJECT_DIR")
    if dir_path is None:
        raise Exception("QBM_PROJECT_DIR env var not set")

    dir_path = Path(dir_path)
    if dir_path.exists():
        return dir_path
    else:
        raise Exception(f"Path '{dir_path}' does not exist")
