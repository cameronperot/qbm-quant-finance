import numpy as np
import pandas as pd

from qbm.utils import convert_bin_list_to_str, split_bin_str


def generate_sample(model, v, n_passes):
    """
    Generate a sample after a number of passes from the provided RBM model from the initial
    state v.

    :param model: Scikitlearn RBM model.
    :param v: Initial state to start the visible layer in.
    :param n_passes: Number of passes through the network to perform, i.e., the number of
        Gibbs sampling steps.

    :returns: The visible layer after sampling.
    """
    for i in range(n_passes):
        v = model.gibbs(v)

    return v


def generate_samples_df(
    model,
    n_samples,
    n_visible,
    n_passes,
    columns,
    split_points=[16, 32, 48],
    sequential=True,
):
    """
    Generate a dataframe of samples.

    :param model: Scikitlearn RBM model.
    :param n_samples: Number of samples to generate.
    :param n_visible: Number of visible units in the model.
    :param n_passes: Number of passes through the network to perform, i.e., the number of
    :param columns: Names of the columns, must match the order of the binary strings.
    :param v: Initial state to start the visible layer in.
        Gibbs sampling steps.
    :param split_points: Points at which to split the binary string.
    :param sequential: Boolean to generate the samples sequentially (one starting random
            input) if True, or in parallel (n_samples starting random inputs) if False.

    :returns: Dataframe of samples.
    """
    samples = []

    if sequential:
        v = np.random.choice([0, 1], n_visible)
        for i in range(n_samples):
            v = generate_sample(model, v, n_passes)
            samples.append(
                map(convert_bin_list_to_str, np.array_split(v, split_points))
            )

    else:
        V = np.random.choice([0, 1], (n_samples, n_visible))
        V = generate_sample(model, V, n_passes)
        for v in V:
            samples.append(
                map(convert_bin_list_to_str, np.array_split(v, split_points))
            )

    return pd.DataFrame(samples, columns=columns)
