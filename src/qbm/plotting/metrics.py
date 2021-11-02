import matplotlib.pyplot as plt


def plot_autocorrelation(ax, lags, acf, title):
    """
    Plots the autocorrelation (acf) as a function of the lag on ax.

    :param ax: Matplotlib axis.
    :param lags: Lags over which the acf is computed.
    :param acf: Autocorrelation function.

    :returns: Matplotlib axis.
    """
    ax.plot(lags, acf)
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 10 ** 4)
    ax.set_ylim(10 ** -4, 1)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.grid(alpha=0.7)

    return ax


def plot_autocorrelation_grid(samples):
    """
    Plots the autocorrelation function (acf) for all currencies in the samples dataframe.

    :param sample: Dataframe of samples.
    :param
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), dpi=300)
    for column, ax in zip(samples.columns, axs.flatten()):
        plot_autocorrelation(ax, range(len(samples[column])), samples[column], column)

    plt.tight_layout()

    return fig, axs


def plot_correlation_coefficients(data, samples):
    """
    Plots the data correlation coefficients against those of the samples.

    :param data: Correlation coefficients of the data.
    :param samples: Correlation coefficients of the samples, this is
        a dict of dataframes, containing keys ["means", "stds"] (e.g. output of
        compute_stats_over_dfs).

    :returns: Matplotlib figure and axes.
    """
    fig, axs = plt.subplots(2, 3, figsize=(10, 6), dpi=300)
    for i, (row, ax) in enumerate(zip(data.index, axs.flatten())):
        ax.set_title(row)
        ax.errorbar(
            range(3),
            samples["means"].loc[row],
            label="Sample Ensemble",
            yerr=samples["stds"].loc[row],
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
        ax.set_xlim((-0.75, 2.75))
        ax.set_xticks(ticks=range(3))
        ax.set_xticklabels(labels=data.columns)
        ax.grid(alpha=0.7)
        if i == 0:
            ax.legend()

    plt.tight_layout()

    return fig, axs


def plot_qq(ax, data, samples, title, params, **kwargs):
    """
    Plots a QQ plot of the data against the samples on the provided ax.

    :param ax: Matplotlib axis.
    :param data: Array of data points.
    :param samples: Array of samples.
    :param title: Title of the plot.
    :param params: Additional parameter dictionary for ax configuration, required keys are
        ["xlims", "ylims", "xticks", "yticks"].

    :returns: Matplotlib axis.
    """
    ax.set_aspect("equal")
    ax.plot(
        params["xlims"],
        params["ylims"],
        color="tab:red",
        alpha=0.7,
    )
    ax.scatter(sorted(data), sorted(samples), **kwargs)
    ax.set_title(title)
    ax.set_xlim(params["xlims"])
    ax.set_ylim(params["ylims"])
    ax.set_xticks(params["xticks"])
    ax.set_yticks(params["yticks"])
    ax.set_xlabel("Data")
    ax.set_ylabel("Samples")
    ax.grid(alpha=0.7)

    return ax


def plot_qq_grid(data, samples, params):
    """
    Plots a 2x2 grid of QQ plots.

    :param data: Data must be a dataframe of shape (N, 4).
    :param samples: samples must be a dataframe with matching column names to the data.
    :param params: Additional parameter dictionary for ax configuration, required keys are
        ["xlims", "ylims", "xticks", "yticks"].

    :returns: Matplotlib figure and axes.
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=300, tight_layout=True)
    for column, ax in zip(data.columns, axs.flatten()):
        plot_qq(ax, data[column], samples[column], column, params)

    plt.tight_layout()

    return fig, axs


def plot_volatilities(data, samples, params):
    """
    Plots the data annualized volatility against those of the samples.

    :param data: Annualized volatility of the data.
    :param samples: Annualized volatility of the samples, this is
        a dict of series, containing keys ["means", "stds"] (e.g. output of
        compute_stats_over_dfs).

    :returns: Matplotlib figure and axes.
    """
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.set_title("Annualized Volatility")
    ax.errorbar(
        range(4),
        samples["means"],
        label="Sample Ensemble",
        yerr=samples["stds"],
        fmt="o",
        markersize=6,
        linewidth=2,
        capsize=12,
        zorder=1,
    )
    ax.scatter(
        range(4),
        data,
        label="Data",
        marker="x",
        c="tab:red",
        s=100,
        zorder=2,
    )
    ax.set_xlim(params["xlims"])
    ax.set_ylim(params["ylims"])
    ax.set_xticks(ticks=range(4))
    ax.set_yticks(params["yticks"])
    ax.set_xticklabels(labels=data.index)
    ax.grid(alpha=0.7)
    ax.legend()

    plt.tight_layout()

    return fig, ax
