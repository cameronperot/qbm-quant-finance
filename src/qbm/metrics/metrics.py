import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau


def compute_correlation_coefficients(df, combinations, sample=None):
    """
    Computes the Pearson r, Spearman r, and Kendall τ coefficients for all combinations
    of columns in the provided dataframe.

    :param df: Dataframe.
    :param sample: If sample is provided, then an additional column will be appended to
        the dataframe corresponding to the sample number.

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
        if sample is not None:
            correlation_coefficients[f"{X}/{Y}"]["sample"] = sample

    return pd.DataFrame.from_dict(correlation_coefficients, orient="index")


def compute_annualized_volatility(df):
    """
    Computes the annualized volatility of the dataframe.

    :param df: Dataframe.

    :returns: Series of the dataframe's volatility.
    """
    return df.std() * np.sqrt(252)
