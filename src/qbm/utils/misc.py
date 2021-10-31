import os
import pickle
import numpy as np
import pandas as pd

from pathlib import Path


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


def _get_path_from_env(env_var):
    """
    Gets the path from the environment and checks if it is valid.

    :param dir_path: String of the env var, e.g. "QBM_ARTIFACTS_DIR" or "QBM_DATA_DIR".

    :returns: Path.
    """
    dir_path = os.getenv(env_var)
    if dir_path is None:
        raise Exception(f"{env_var} env var not set")

    dir_path = Path(dir_path)
    if dir_path.exists():
        return dir_path
    else:
        raise Exception(f"Path '{dir_path}' does not exist")
