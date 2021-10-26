import matplotlib.pyplot as plt


def plot_correlation_coefficients(
    correlation_coefficients_data, correlation_coefficients_samples
):
    """
    Plots the data correlation coefficients against those of samples.

    :param correlation_coefficients_data: Correlation coefficients of the data.
    :param correlation_coefficients_samples: Correlation coefficients of the data, this is
        a dict of dataframes, containing keys ["means", "stds"] (e.g. output of
        compute_stats_over_dfs).

    :returns: Matplotlib figure and axes.
    """
    fig, axs = plt.subplots(2, 3, figsize=(10, 6), dpi=300)
    for i, (row, ax) in enumerate(
        zip(correlation_coefficients_data.index, axs.flatten())
    ):
        ax.set_title(row)
        ax.grid(alpha=0.7, zorder=0)
        ax.errorbar(
            range(3),
            correlation_coefficients_samples["means"].loc[row],
            label="Sample Ensemble",
            yerr=correlation_coefficients_samples["stds"].loc[row],
            fmt="o",
            markersize=4,
            linewidth=1.8,
            capsize=6,
            zorder=1,
        )
        ax.scatter(
            range(3),
            correlation_coefficients_data.loc[row],
            label="Data",
            marker="x",
            c="r",
            s=32,
            zorder=2,
            alpha=0.7,
        )
        ax.set_xlim((-0.75, 2.75))
        ax.set_xticks(ticks=range(3))
        ax.set_xticklabels(labels=correlation_coefficients_data.columns)
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
        color="r",
        alpha=0.7,
    )
    ax.scatter(sorted(data), sorted(samples), **kwargs)
    ax.set_title(title)
    ax.set_xlim(params["xlims"])
    ax.set_ylim(params["ylims"])
    ax.set_xticks(params["xticks"])
    ax.set_yticks(params["yticks"])
    ax.grid(alpha=0.7)

    return ax


def plot_qq_grid(data, samples, params):
    """
    Plots a 2x2 grid of QQ plots.

    :param data: Data must be a dataframe of shape (N, 4).
    :param samples: Samples must be a dataframe with matching column names to the data.
    :param params: Additional parameter dictionary for ax configuration, required keys are
        ["xlims", "ylims", "xticks", "yticks"].

    :returns: Matplotlib figure and axes.
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=300, tight_layout=True)
    for dataset, ax in zip(data.columns, axs.flatten()):
        plot_qq(ax, data[dataset], samples[dataset], dataset, params)

    return fig, axs
