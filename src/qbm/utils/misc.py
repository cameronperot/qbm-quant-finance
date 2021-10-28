import os
import pickle
import pandas as pd

from pathlib import Path


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


def save_artifact(artifact, file_name):
    """
    Saves an artifact to the artifacts directory specified by the QBM_ARTIFACTS_DIR env var.

    :param artifact: Python object to save.
    :param file_name: Name of the file to save (will be suffixed with .pkl).
    """
    artifacts_dir = _get_path_from_env("QBM_ARTIFACTS_DIR")
    with open(artifacts_dir / f"{file_name}.pkl", "wb") as f:
        pickle.dump(artifact, f)


def load_artifact(file_name):
    """
    Loads an artifact from the artifacts directory specified by the QBM_ARTIFACTS_DIR env var.

    :param file_name: Name of the file to load (will be suffixed with .pkl).

    :returns: Loaded python object.
    """
    artifacts_dir = _get_path_from_env("QBM_ARTIFACTS_DIR")
    with open(artifacts_dir / f"{file_name}.pkl", "rb") as f:
        return pickle.load(f)


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
