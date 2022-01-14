import json
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
    return pd.DataFrame.from_dict(
        {
            "min": df.min(),
            "max": df.max(),
            "mean": df.mean(),
            "median": df.median(),
            "std": df.std(),
        },
        orient="index",
    )


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


def compute_lower_tail_concentration(z, U, V):
    """
    Lower tail concentration function defined as:
    L(z) = P(U <= z | V <= z) = P(U <= z, V <= z) / P(U <= z)
    References:
        - https://freakonometrics.hypotheses.org/2435
        - https://openacttexts.github.io/Loss-Data-Analytics/C-DependenceModel
            (section 14.5.4.3)
        - https://www.casact.org/sites/default/files/old/studynotes_venter_tails_of_copulas.pdf
            (section 3)

    :param z: Tail dependence parameter.
    :param U: Input array for first variable (e.g. X.rank() / (len(X) + 1)).
    :param V: Input array for second variable (e.g. Y.rank() / (len(Y) + 1)).

    :returns: Lower tail concentration function.
    """
    return np.sum(np.logical_and(U <= z, V <= z)) / np.sum(U <= z)


compute_lower_tail_concentration = np.vectorize(
    compute_lower_tail_concentration, excluded=[1, 2]
)


def compute_upper_tail_concentration(z, U, V):
    """
    Upper tail concentration function defined as:
    R(z) = P(U > z | V > z) = P(U > z, V > z) / P(U > z)
    References:
        - https://freakonometrics.hypotheses.org/2435
        - https://openacttexts.github.io/Loss-Data-Analytics/C-DependenceModel
            (section 14.5.4.3)
        - https://www.casact.org/sites/default/files/old/studynotes_venter_tails_of_copulas.pdf
            (section 3)

    :param z: Tail dependence parameter.
    :param U: Input array for first variable (e.g. X.rank() / (len(X) + 1)).
    :param V: Input array for second variable (e.g. Y.rank() / (len(Y) + 1)).

    :returns: Upper tail concentration function.
    """
    return np.sum(np.logical_and(U > z, V > z)) / np.sum(U > z)


compute_upper_tail_concentration = np.vectorize(
    compute_upper_tail_concentration, excluded=[1, 2]
)


def filter_df_on_values(df, column_values, drop_filter_columns=True):
    """
    Return a copy of the dataframe filtered conditionally on provided
    column values.

    :param df: Dataframe to filter.
    :param column_values: Dictionary where the keys are column names, and the
        values are values on which to filter the dataframe.
    :param drop_filter_columns: If True returns a copy of the dataframe with
        the filtered columns dropped.

    :returns: A dataframe filtered conditionally on the provided column values.
    """
    df = df.copy()
    for column, value in column_values.items():
        df = df.loc[df[column] == value]

    if drop_filter_columns:
        df.drop(column_values.keys(), axis=1, inplace=True)

    return df


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


def get_rng(seed):
    """
    Creates a random number generator with the specified seed value.

    :param seed: Seed value for the rng.

    :returns: Numpy RandomState object.
    """
    return RandomState(MT19937(SeedSequence(seed)))


def load_artifact(file_path):
    """
    Loads a pickle or json artifact (depending on the file extension).

    :param file_path: Path of the file to load.

    :returns: Loaded python object.
    """
    if type(file_path) == str:
        file_path = Path(file_path)

    if not file_path.exists() or file_path.suffix not in (".json", ".pkl"):
        raise Exception(f"File {file_path} does not exist")

    if file_path.suffix == ".json":
        with open(file_path, "r") as f:
            return json.load(f)
    elif file_path.suffix == ".pkl":
        with open(file_path, "rb") as f:
            return pickle.load(f)


@np.vectorize
def log_logistic(x):
    """
    Computes the log of the logistic sigmoid function.

    :param x: Value or array of values for which to compute using.

    :returns: log(1 / (1 + e^-x))
    """
    if x > 0:
        return -np.log(1 + np.exp(-x))
    else:
        return x - np.log(1 + np.exp(x))


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


def save_artifact(artifact, file_path):
    """
    Saves a pickle or json artifact (depending on the file extension).

    :param artifact: Python object to save.
    :param file_path: Path of the file to save.
    """
    if type(file_path) == str:
        file_path = Path(file_path)

    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)

    if file_path.suffix not in (".json", ".pkl"):
        raise Exception("Invalid file extension")

    if file_path.suffix == ".json":
        with open(file_path, "w") as f:
            json.dump(artifact, f, indent=4)
    elif file_path.suffix == ".pkl":
        with open(file_path, "wb") as f:
            pickle.dump(artifact, f)
