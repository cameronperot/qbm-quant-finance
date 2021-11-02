import numpy as np


@np.vectorize
def binarize(x, n_bits, x_min, x_max):
    """
    Convert the value x into a n-bit binary string.

    :param x: Value which to convert.
    :param column: Name of the feature to convert w.r.t.
    :param params: Dict containing the number of bits, and entries corresponding to the column
        min and max values (including the ϵ factor).

    :returns: A binary string representation of x.
    """
    scaling_factor = (2 ** n_bits - 1) / (x_max - x_min)

    x = round((x - x_min) * scaling_factor)
    assert x >= 0 and x <= 2 ** n_bits - 1
    return bin(x)[2:].zfill(n_bits)


@np.vectorize
def unbinarize(x, n_bits, x_min, x_max):
    """
    Convert the value x into a float from a n-bit binary string.

    :param x: Value which to convert.
    :param params: Dict containing the number of bits, and entries corresponding to the column
        min and max values (including the ϵ factor).

    :returns: A float representation of x.
    """
    scaling_factor = (2 ** n_bits - 1) / (x_max - x_min)

    assert len(x) == n_bits
    return int(x, 2) / scaling_factor + x_min


def binarize_df(df, params):
    """
    Convert all columns of a dataframe to binary representation.

    :param df: Dataframe which to convert.
    :param params: Dict containing the number of bits, and entries corresponding to the column
        min and max values (including the ϵ factor).

    :returns: A binarized version of df.
    """
    df_binarized = df.copy()
    for column in df.columns:
        df_binarized[column] = binarize(df[column], **params[column])

    return df_binarized


def binarize_volatility(volatility):
    """
    Binarizes the volatilities. Value is 1 if the rolling volatility is greater than
    the historical median, and 0 otherwise.
    """
    volatility_binarized = volatility.copy()
    for column in volatility.columns:
        volatility_binarized[column] = (
            volatility[column] > volatility[column].median()
        ).astype("int8")

    return volatility_binarized


def unbinarize_df(df, params):
    """
    Convert all columns of a dataframe to floats from binary representation.

    :param df: Dataframe which to convert.
    :param params: Dict containing the number of bits, and entries corresponding to the column
        min and max values (including the ϵ factor).

    :returns: An unbinarized version of df_binarized.
    """
    df_unbinarized = df.copy()
    for column in df.columns:
        df_unbinarized[column] = unbinarize(df[column], **params[column])

    return df_unbinarized


def convert_bin_str_to_list(bin_str):
    """
    Converts a binary string to a list of integers.

    :param bin_str: Binary string, e.g. "01100101".

    :returns: List of integers, e.g. [0, 1, 1, 0, 0, 1, 0, 1]
    """
    return np.fromiter((int(x) for x in bin_str), np.int8)


def convert_bin_list_to_str(bin_list):
    """
    Converts a list of integers to a binary string.

    :param bin_list: List of integers, e.g. [0, 1, 1, 0, 0, 1, 0, 1]

    :returns: Binary string, e.g. "01100101".
    """
    return "".join(str(x) for x in bin_list)


def convert_binarized_df_to_input_array(df):
    """
    Converts a dataframe of binary strings to an (N, d) array of integers.

    :param df: Dataframe of binary strings.

    :returns: Numpy array of integers.
    """
    return np.concatenate(
        [
            np.stack(df.applymap(convert_bin_str_to_list)[column])
            for column in df.columns
        ],
        axis=1,
    )


def split_bin_str(bin_str, split_points):
    """
    Splits a binary string into a list of binary strings at the split points. E.g. to split
    a length 64 string into four equal length 16 strings, one would pass split points of
    [0, 16, 32, 48].

    :param bin_str: Binary string.
    :param split_points: List of points at which to split the string.

    :returns: List of split strings.
    """
    return [
        bin_str[start:stop]
        for start, stop in zip(split_points, split_points[1:] + [None])
    ]
