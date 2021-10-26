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

    return v.astype("int64")


def generate_samples_df(
    model, columns, n_samples, n_passes, split_points=[0, 16, 32, 48]
):
    """
    Generate a dataframe of samples.

    :param model: Scikitlearn RBM model.
    :param columns: Names of the columns, must match the order of the binary strings.
    :param n_samples: Number of samples to generate.
    :param v: Initial state to start the visible layer in.
    :param n_passes: Number of passes through the network to perform, i.e., the number of
        Gibbs sampling steps.
    :param split_points: Points at which to split the binary string.

    :returns: Dataframe of samples.
    """
    samples = []
    for i in range(n_samples):
        v = np.random.choice([0, 1], data.shape[1])
        v = generate_sample(model, v, n_passes)
        v = convert_bin_list_to_str(v)
        samples.append(split_bin_str(v, split_points))

    return pd.DataFrame(samples, columns=columns)
