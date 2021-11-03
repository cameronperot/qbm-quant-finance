import numpy as np


@np.vectorize
def binarize(x, n_bits, x_min, x_max, **kwargs):
    """
    Convert the value x into a n-bit binary string.

    :param x: Value which to convert.
    :param n_bits: Length of the binary string.
    :param x_min: Minimum value for scaling.
    :param x_max: Maximum value for scaling.

    :returns: A binary string representation of x.
    """
    scaling_factor = (2 ** n_bits - 1) / (x_max - x_min)

    x = round((x - x_min) * scaling_factor)
    assert x >= 0 and x <= 2 ** n_bits - 1
    return bin(x)[2:].zfill(n_bits)


def binarize_df(df, binarization_params):
    """
    Convert all columns of a dataframe to binary representation.

    :param df: Dataframe which to convert.
    :param binarization_params: Dict containing the number of bits, and entries corresponding
        to the column min and max values (including the ϵ factor).

    :returns: A binarized version of df.
    """
    df_binarized = df.copy()
    for column in df.columns:
        df_binarized[column] = binarize(df[column], **binarization_params[column])

    return df_binarized


def binarize_volatility(volatility):
    """
    Binarizes the volatilities. Value is 1 if the rolling volatility is greater than the
        historical median, and 0 otherwise.

    :param volatility: Dataframe of rolling volatilities.

    :returns: Dataframe of binarized rolling volatilities.
    """
    volatility_binarized = volatility.copy()
    for column in volatility.columns:
        volatility_binarized[column] = (
            volatility[column] > volatility[column].median()
        ).astype(np.int8)

    # append "_binary" to the column names
    column_map = {column: f"{column}_binary" for column in volatility.columns}
    volatility_binarized.rename(columns=column_map, inplace=True)

    return volatility_binarized


def convert_bin_list_to_str(bin_list):
    """
    Converts a list of integers to a binary string.

    :param bin_list: List of binary numbers, e.g. [0, 1, 1, 0, 0, 1, 0, 1]

    :returns: Binary string, e.g. "01100101".
    """
    return "".join(str(x) for x in bin_list)


def convert_bin_str_to_list(bin_str):
    """
    Converts a binary string to a list of integers.

    :param bin_str: Binary string, e.g. "01100101".

    :returns: List of integers, e.g. [0, 1, 1, 0, 0, 1, 0, 1]
    """
    return np.fromiter((int(x) for x in bin_str), np.int8)


@np.vectorize
def unbinarize(x, n_bits, x_min, x_max, **kwargs):
    """
    Convert the value x into a float from a n-bit binary string.

    :param x: Value which to convert.
    :param n_bits: Length of the binary string.
    :param x_min: Minimum value for scaling.
    :param x_max: Maximum value for scaling.

    :returns: A float representation of x.
    """
    scaling_factor = (2 ** n_bits - 1) / (x_max - x_min)

    assert len(x) == n_bits
    return int(x, 2) / scaling_factor + x_min


def unbinarize_df(df, binarization_params):
    """
    Convert all columns of a dataframe to floats from binary representation.

    :param df: Dataframe which to convert.
    :param binarization_params: Dict containing the number of bits, and entries corresponding
        to the column min and max values (including the ϵ factor).

    :returns: An unbinarized version of df_binarized.
    """
    df_unbinarized = df.copy()
    for column in df.columns:
        if column.endswith("_binary"):
            df_unbinarized[column] = df[column].astype(np.int8)
        else:
            df_unbinarized[column] = unbinarize(
                df[column], **binarization_params[column]
            )

    return df_unbinarized
