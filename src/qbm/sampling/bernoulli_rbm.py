import numpy as np
import pandas as pd


def generate_rbm_sample(model, v, n_steps):
    """
    Generate a sample after a number of passes from the provided RBM model from the initial
    state v.

    :param model: Scikitlearn RBM model.
    :param v: Initial state to start the visible layer in.
    :param n_steps: Number of passes through the network to perform, i.e., the number of
        Gibbs sampling steps.

    :returns: The visible layer after n_steps Gibbs sampling steps.
    """
    for i in range(n_steps):
        v = model.gibbs(v)

    return v.astype(np.int8)


def generate_rbm_samples_df(
    model, v, n_samples, n_steps, model_params,
):
    """
    Generate a dataframe of samples.

    :param model: Scikitlearn RBM model.
    :param v: Initial state to start the visible layer in.
    :param n_samples: Number of samples to generate.
    :param n_steps: Number of passes through the network to perform, i.e., the number of
        Gibbs sampling steps.
    :param model_params: Dict with required keys
        ["n_bits", "columns", "binarization_params", "split_indices"]

    :returns: Dataframe of samples, final visible state.
    """
    n_bits = model_params["n_bits"]
    columns = model_params["columns"]
    split_indices = model_params["split_indices"]

    samples = np.empty((n_samples, len(columns)), f"U{n_bits}")

    # generate samples sequentially from a single initial visible layer
    if v.ndim == 1:
        for i in range(n_samples):
            v = generate_rbm_sample(model, v, n_steps)
            samples[i] = np.stack(
                ["".join(x) for x in np.array_split(v.astype("str"), split_indices)]
            )

    # generate samples in parallel from multiple initial visible layers
    elif v.ndim == 2:
        assert v.shape[0] == n_samples
        v = generate_rbm_sample(model, v, n_steps)
        for i in range(n_samples):
            samples[i] = np.stack(
                ["".join(x) for x in np.array_split(v[i].astype("str"), split_indices)]
            )

    return pd.DataFrame(samples, columns=columns), v
