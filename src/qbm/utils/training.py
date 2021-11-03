import numpy as np

from qbm.utils import convert_bin_str_to_list


def prepare_training_data(log_returns_binarized, additional_binary_variables=None):
    """
    Converts a dataframe of binary strings to an array of binary numbers of shape
    (n_samples, n_visible).

    :param log_returns_binarized: Dataframe of binary strings.
    :param additional_binary_variables: Optional dataframe of binary numbers.

    :returns: Dictionary with keys ["X_train", "columns", "split_indices", "index"].
    """
    if additional_binary_variables is not None:
        # ensure the indices line up
        log_returns_binarized = log_returns_binarized.loc[
            additional_binary_variables.index
        ]
        assert (additional_binary_variables.index == log_returns_binarized.index).all()

        # ensure proper dtype for additional binary variables
        assert (additional_binary_variables.dtypes == np.int8).all()

    # set variable with column name ordering
    columns = log_returns_binarized.columns.to_list()

    # convert all entries from binary strings to arrays of binary integers
    log_returns_binarized = log_returns_binarized.applymap(convert_bin_str_to_list)

    # stack the columns into numpy arrays and compute the split points
    X_train = []
    split_indices = []
    for i, column in enumerate(columns):
        x = np.stack(log_returns_binarized[column])
        X_train.append(x)
        split_indices.append(
            x.shape[1] if i == 0 else split_indices[i - 1] + x.shape[1]
        )

    # concatenate the columns into a single array
    X_train = np.concatenate(X_train, axis=1)

    if additional_binary_variables is not None:
        # concatenate the additonal binary variables onto the log returns array
        X_train = np.concatenate([X_train, additional_binary_variables], axis=1)

        # update the column names list and split points
        columns += additional_binary_variables.columns.to_list()
        for i in range(additional_binary_variables.shape[1]):
            split_indices.append(split_indices[-1] + 1)

    return {
        "X_train": X_train,
        "columns": columns,
        "split_indices": split_indices[:-1],
        "index": log_returns_binarized.index,
    }
