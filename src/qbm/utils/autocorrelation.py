import numpy as np
from numba import njit


@njit(boundscheck=True)
def compute_bin_averages(x, bin_size):
    """
    Computes the bin averages, used in computing the autocorrelation time.

    :param x: Array of samples.
    :param n: Number of samples per bin.

    :returns: Bin averages.
    """
    n_samples = len(x)
    n_bins = int(n_samples / bin_size)
    bin_avgs = np.zeros(n_bins)

    for k in range(n_bins):
        a = k * bin_size
        b = (k + 1) * bin_size
        bin_avgs[k] = x[a:b].mean()

    return bin_avgs


@njit(boundscheck=True)
def compute_autocorrelation_time(x, bin_sizes):
    """
    Computes the quantity Nε^2 / (2σ^2) using the binning method. The
    autocorrelation time is the value at which the output sequence asymptotes at.

    :param x: Array of samples.
    :param bin_sizes: List of bin sizes.

    :returns: Bin averages to be plotted against ns.
    """
    n_samples = len(x)
    result = np.zeros(len(bin_sizes))

    for i, bin_size in enumerate(bin_sizes):
        n_bins = int(n_samples / bin_size)
        bin_avgs = compute_bin_averages(x, bin_size)
        ε_sq = ((bin_avgs - bin_avgs.mean()) ** 2).sum() / (n_bins * (n_bins - 1))
        result[i] = n_samples * ε_sq / (2 * x.var())

    return result


def iat_BM(data):
    """
    Batch means integrated autocorrelation time,
    Thompson (2010) Eq (2)
    """
    b_s = int(len(data) ** (1.0 / 3.0))
    data_split = np.array_split(data, b_s)
    b_means = []
    for batch in data_split:
        b_means.append(np.mean(batch))
    var_b = np.var(b_means)
    var_d = np.var(data)
    iat = len(data_split[0]) * var_b / var_d
    return iat
