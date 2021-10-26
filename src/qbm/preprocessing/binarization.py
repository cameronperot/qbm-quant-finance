import numpy as np
import pandas as pd


@np.vectorize
def binarize(x, x_min, x_max, n_bits):
    """
    Convert the value x into a n-bit binary string.

    :param x: Value which to convert.
    :param x_min: Minimum value of the corresponding series.
    :param x_min: Maximum value of the corresponding series.
    :param n_bits: Number of bits in the binary string.

    :returns: A binary string representation of x.
    """
    scaling_factor = (2 ** n_bits - 1) / (x_max - x_min)

    x = round((x - x_min) * scaling_factor)
    assert x >= 0 and x <= 2 ** n_bits - 1
    return bin(x)[2:].zfill(n_bits)


@np.vectorize
def unbinarize(x_binarized, x_min, x_max, n_bits):
    """
    Convert the value x_bin into a float from a n-bit binary string.

    :param x_binarized: Value which to convert.
    :param x_min: Minimum value of the corresponding series.
    :param x_min: Maximum value of the corresponding series.
    :param n_bits: Number of bits in the binary string.

    :returns: A float representation of x.
    """
    scaling_factor = (2 ** n_bits - 1) / (x_max - x_min)

    assert len(x_binarized) == n_bits
    return int(x_binarized, 2) / scaling_factor + x_min


def binarize_df(df, n_bits, ϵ_min=0, ϵ_max=0):
    """
    Convert all columns of a dataframe to binary representation.

    :param df: Dataframe which to convert.
    :param n_bits: Number of bits in the binary string.
    :param ϵ_min: ϵ tolerance on the minimum end.
    :param ϵ_max: ϵ tolerance on the maximum end.

    :returns: A binarized version of df.
    """
    df_binarized = df.copy()
    for column in df.columns:
        df_binarized[column] = binarize(
            df[column], df[column].min() - ϵ_min, df[column].max() + ϵ_max, n_bits
        )

    return df_binarized


def unbinarize_df(df_binarized, df, n_bits, ϵ_min=0, ϵ_max=0):
    """
    Convert all columns of a dataframe to floats from binary representation.

    :param df_binarized: Dataframe which to convert.
    :param n_bits: Number of bits in the binary string.
    :param ϵ_min: ϵ tolerance on the minimum end.
    :param ϵ_max: ϵ tolerance on the maximum end.

    :returns: An unbinarized version of df_binarized.
    """
    df_unbinarized = df.copy()
    for column in df.columns:
        df_unbinarized[column] = unbinarize(
            df_binarized[column],
            df[column].min() - ϵ_min,
            df[column].max() + ϵ_max,
            n_bits,
        )

    return df_unbinarized


def convert_bin_str_to_list(bin_str):
    """
    Converts a binary string to a list of integers.

    :param bin_str: Binary string, e.g. "01100101".

    :returns: List of integers, e.g. [0, 1, 1, 0, 0, 1, 0, 1]
    """
    return np.array([int(x) for x in bin_str])


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
