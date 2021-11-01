import numpy as np
import pandas as pd


def generate_sample(model, v, n_steps):
    """
    Generate a sample after a number of passes from the provided RBM model from the initial
    state v.

    :param model: Scikitlearn RBM model.
    :param v: Initial state to start the visible layer in.
    :param n_steps: Number of passes through the network to perform, i.e., the number of
        Gibbs sampling steps.

    :returns: The visible layer after sampling.
    """
    for i in range(n_steps):
        v = model.gibbs(v)

    return v.astype("int8")


def generate_samples_df(
    model,
    v,
    n_samples,
    n_steps,
    columns,
    binarization_params,
    split_points=[16, 32, 48],
):
    """
    Generate a dataframe of samples.

    :param model: Scikitlearn RBM model.
    :param v: Initial state to start the visible layer in.
    :param n_samples: Number of samples to generate.
    :param n_steps: Number of passes through the network to perform, i.e., the number of
        Gibbs sampling steps.
    :param columns: Names of the columns, must match the order of the binary strings.
    :param binarization_params: Parameters for (un)binarizing the data.
    :param split_points: Points at which to split the binary string.

    :returns: Dataframe of samples, final visible state.
    """
    samples = np.empty(
        (n_samples, len(columns)), f"U{binarization_params[columns[0]]['n_bits']}"
    )

    if v.ndim == 1:
        for i in range(n_samples):
            v = generate_sample(model, v, n_steps)
            samples[i] = np.stack(
                ["".join(x) for x in np.array_split(v.astype("str"), split_points)]
            )

    elif v.ndim == 2:
        assert v.shape[0] == n_samples
        v = generate_sample(model, v, n_steps)
        for i in range(n_samples):
            samples[i] = np.stack(
                ["".join(x) for x in np.array_split(v[i].astype("str"), split_points)]
            )

    return pd.DataFrame(samples, columns=columns), v
