import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau


def compute_annualized_volatility(df):
    """
    Computes the annualized volatility of the dataframe.

    :param df: Dataframe.

    :returns: Series of the dataframe's volatility.
    """
    return df.std() * np.sqrt(252)


def compute_correlation_coefficients(df, combinations, sample_number=None):
    """
    Computes the Pearson r, Spearman r, and Kendall τ coefficients for all combinations
    of columns in the provided dataframe.

    :param df: Dataframe.
    :param combinations: List of tuples of column name pairs to compute the correlations of.
    :param sample_number: If sample number is provided, then an additional column will be
        appended to the dataframe corresponding to the sample number.

    :returns: Dataframe where rows are the combinations and columns are the correlation
        coefficients.
    """
    correlation_coefficients = {}
    for (X, Y) in combinations:
        correlation_coefficients[f"{X}/{Y}"] = {
            "Pearson": pearsonr(df[X], df[Y])[0],
            "Spearman": spearmanr(df[X], df[Y])[0],
            "Kendall": kendalltau(df[X], df[Y])[0],
        }

        if sample_number is not None:
            correlation_coefficients[f"{X}/{Y}"]["sample"] = sample_number

    return pd.DataFrame.from_dict(correlation_coefficients, orient="index")


def compute_rolling_volatility(df, window, drop_insufficient_data_rows=True):
    """
    Compute the rolling volatility of the input dataframe over the provided
    window.

    :param df: Dataframe of which to compute the volatility from.
    :param window: Window over which to compute the volatility, e.g. timdelta(days=90).
    :param drop_insufficient_data_rows: If True will drop rows that do not have enough data
        to fill the window.

    :returns: Dataframe of the rolling volatilities.
    """
    # compute the rolling volatility
    df = df.rolling(window).apply(compute_annualized_volatility)

    if drop_insufficient_data_rows:
        # drop data without a large enough window
        df = df.loc[df.index >= df.index.min() + window]

    # append "_volatility" to the column names
    column_map = {column: f"{column}_volatility" for column in df.columns}
    df.rename(columns=column_map, inplace=True)

    return df
