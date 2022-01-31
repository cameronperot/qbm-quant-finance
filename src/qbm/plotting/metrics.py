import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter
from datetime import timedelta

from qbm.utils import compute_lower_tail_concentration, compute_upper_tail_concentration


def plot_autocorrelation(ax, lags, acf, title, **kwargs):
    """
    Plots the autocorrelation (acf) as a function of the lag on ax.

    :param ax: Matplotlib axis.
    :param lags: Lags over which the acf is computed.
    :param acf: Autocorrelation function.
    :param title: Title of the plot.

    :returns: Matplotlib axis.
    """
    ax.plot(lags, acf, **kwargs)
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 10 ** 4)
    ax.set_ylim(10 ** -4, 1)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.grid(alpha=0.7)

    return ax


def plot_autocorrelation_grid(acfs, colors=None):
    """
    Plots the autocorrelation function (acf) for all currencies in the sample dataframe.

    :param acfs: Dataframe of autocorrelation function values, or dictionary of dataframes,
        where the keys are the labels.

    :returns: Matplotlib figure and axes.
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), dpi=300)

    if isinstance(acfs, pd.DataFrame):
        for column, ax in zip(acfs.columns, axs.flatten()):
            lags = acfs.index
            plot_autocorrelation(ax, lags, acfs[column], title=column, color=colors[0])

    elif isinstance(acfs, dict):
        acfs_dict = acfs.copy()
        labels = list(acfs_dict.keys())
        columns = acfs_dict[labels[0]].columns
        for i, (column, ax) in enumerate(zip(columns, axs.flatten())):
            for j, (label, acfs) in enumerate(acfs_dict.items()):
                lags = acfs.index
                if j == 0:
                    ax = plot_autocorrelation(
                        ax,
                        lags,
                        acfs[column],
                        title=column,
                        label=label,
                        color=colors[label],
                    )
                else:
                    ax.plot(lags, acfs[column], label=label, color=colors[label])

            if i == 0:
                ax.legend(loc="lower left")

    plt.tight_layout()

    return fig, axs


def plot_correlation_coefficients(data, sample):
    """
    Plots the data correlation coefficients against those of the sample.

    :param data: Correlation coefficients of the data.
    :param sample: Correlation coefficients of the sample, this is
        a dict of dataframes, containing keys ["means", "stds"] (e.g. output of
        compute_stats_over_dfs).

    :returns: Matplotlib figure and axes.
    """
    fig, axs = plt.subplots(2, 3, figsize=(10, 6), dpi=300)
    for i, (row, ax) in enumerate(zip(data.index, axs.flatten())):
        ax.set_title(row)
        ax.errorbar(
            range(3),
            sample["means"].loc[row],
            label="Sample Ensemble",
            yerr=sample["stds"].loc[row],
            fmt="o",
            markersize=4,
            linewidth=1.8,
            capsize=6,
            zorder=1,
        )
        ax.scatter(
            range(3),
            data.loc[row],
            label="Data",
            marker="x",
            c="tab:red",
            s=36,
            zorder=2,
            alpha=0.7,
        )
        ax.set_xticks(ticks=range(3))
        ax.set_xticklabels(labels=data.columns)
        ax.set_xlim((-0.75, 2.75))
        ax.grid(alpha=0.7)
        if i == 0:
            ax.legend(loc="lower left")

    plt.tight_layout()

    return fig, axs


def plot_qq(ax, data, sample, title, params, **kwargs):
    """
    Plots a QQ plot of the data against the sample on the provided ax.

    :param ax: Matplotlib axis.
    :param data: Array of data points.
    :param sample: Array of sample.
    :param title: Title of the plot.
    :param params: Additional parameter dictionary for ax configuration, required keys are
        ["xlims", "ylims", "xticks", "yticks"].

    :returns: Matplotlib axis.
    """
    ax.set_aspect("equal")
    ax.plot(
        params["xlims"], params["ylims"], color="tab:red", alpha=0.7,
    )
    ax.scatter(sorted(data), sorted(sample), **kwargs)
    ax.set_title(title)
    ax.set_xticks(params["xticks"])
    ax.set_yticks(params["yticks"])
    ax.set_xlim(params["xlims"])
    ax.set_ylim(params["ylims"])
    ax.set_xlabel("Data")
    ax.set_ylabel("Model")
    ax.grid(alpha=0.7)

    return ax


def plot_qq_grid(data, sample, params):
    """
    Plots a 2x2 grid of QQ plots.

    :param data: A dataframe of shape (n_sample, 4).
    :param sample: A dataframe with matching column names to the data and the same shape.
    :param params: Additional parameter dictionary for ax configuration, required keys are
        ["xlims", "ylims", "xticks", "yticks"].

    :returns: Matplotlib figure and axes.
    """
    fig, axs = plt.subplots(2, 2, figsize=(9, 9), dpi=300, tight_layout=True)
    for column, ax in zip(data.columns, axs.flatten()):
        plot_qq(ax, data[column], sample[column], column, params)

    plt.tight_layout()

    return fig, axs


def plot_volatility_comparison(data, sample, params):
    """
    Plots the data annualized volatility against those of the sample on an error bar plot.

    :param data: Annualized volatility of the data.
    :param sample: Annualized volatility of the sample, this is a dict of series,
        containing keys ["means", "stds"] (e.g. output of compute_stats_over_dfs).
    :param params: Additional parameter dictionary for ax configuration, required keys are
        ["xlims", "ylims", "yticks"].

    :returns: Matplotlib figure and axes.
    """
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.set_title("Annualized Volatility")
    ax.errorbar(
        range(len(data)),
        sample["means"],
        label="Sample Ensemble",
        yerr=sample["stds"],
        fmt="o",
        markersize=6,
        linewidth=2,
        capsize=12,
        zorder=1,
    )
    ax.scatter(
        range(len(data)), data, label="Data", marker="x", c="tab:red", s=100, zorder=2,
    )
    ax.set_xticks(ticks=range(len(data)))
    ax.set_yticks(params["yticks"])
    ax.set_xticklabels(labels=data.index)
    ax.set_xlim(params["xlims"])
    ax.set_ylim(params["ylims"])
    ax.grid(alpha=0.7)
    ax.legend()

    plt.tight_layout()

    return fig, ax


def plot_volatility(ax, dates, volatility, title, params):
    """
    Plots the historical volatility against the date, as well as a horizontal line indicating
    the median value over the entire historical period.

    :param ax: Matplotlib axis.
    :param dates: Array of dates to plot against.
    :param volatility: Array of volatility values to plot against the dates.
    :param title: Title of the plot.
    :param params: Additional parameter dictionary for ax configuration, required keys are
        ["xlims", "ylims", "xticks", "yscale", "label"].

    :returns: Matplotlib axis.
    """
    ax.plot(dates, volatility, label=params["label"], zorder=0)
    ax.hlines(
        volatility.median(),
        dates.min() - timedelta(weeks=100),
        dates.max() + timedelta(weeks=100),
        label="Median",
        linewidth=2,
        color="tab:red",
        zorder=1,
    )
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.set_xticks(params["xticks"])
    ax.set_yscale(params["yscale"])
    ax.set_xlim(params["xlims"])
    ax.set_ylim(params["ylims"])
    ax.xaxis.grid(alpha=0.7)
    ax.yaxis.grid(False)
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))


def plot_tail_concentrations_grid(dfs, combinations, colors, interval_size=1e-3):
    """
    Plots the upper and lower tail concentration functions for the provided
    dataframes.

    :param dfs: Dictionary where keys are the plot labels, and the values are the
        corresponding dataframes to be plotted.
    :param combinations: List of tuples of column names to be plotted, e.g.
        [("EURUSD", "GBPUSD"), ("EURUSD", "USDJPY"), ...]
    :param colors: Dictionary where the keys are the plot labels, and the values are colors.
    :param interval_size: Spacing between the values on the independent axis.

    :returns: Matplotlib figure and axes.
    """
    fig, axs = plt.subplots(3, 2, figsize=(11, 11), dpi=300)

    for i, (ax, (pair_X, pair_Y)) in enumerate(zip(axs.flatten(), combinations)):
        for j, (label, df) in enumerate(dfs.items()):
            U = df[pair_X].rank() / (df.shape[0] + 1)
            V = df[pair_Y].rank() / (df.shape[0] + 1)

            z_lower = np.arange(interval_size, 0.5 + interval_size, interval_size)
            z_upper = np.arange(0.5, 1, interval_size)
            lower_concentration = compute_lower_tail_concentration(z_lower, U, V)
            upper_concentration = compute_upper_tail_concentration(z_upper, U, V)

            x = np.concatenate([z_lower, z_upper])
            y = np.concatenate([lower_concentration, upper_concentration])

            ax.plot(x, y, label=label, color=colors[label])
            ax.set_title(f"{pair_X}/{pair_Y}")
            ax.grid(True)

        if i == 0:
            ax.legend(loc="lower center")

    plt.tight_layout()

    return fig, axs


def plot_volatility_grid(volatility, params):
    """
    Plots a 2x2 grid of the historical volatilities in the provided dataframe.

    :param volatility: Dataframe of historical volatilities.
    :param params: Additional parameter dictionary for ax configuration, required keys are
        ["xlims", "ylims", "xticks", "yscale", "label"].

    :returns: Matplotlib figure and axes.
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), dpi=300)
    for column, ax in zip(volatility.columns, axs.flatten()):
        plot_volatility(
            ax, volatility.index, volatility[column], column.split("_")[0], params
        )
    axs[0, 0].legend()
    plt.tight_layout()

    return fig, axs
