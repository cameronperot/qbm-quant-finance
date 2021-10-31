import matplotlib.pyplot as plt


def plot_correlation_coefficients(
    correlation_coefficients_data, correlation_coefficients_generated
):
    """
    Plots the data correlation coefficients against those of generated.

    :param correlation_coefficients_data: Correlation coefficients of the data.
    :param correlation_coefficients_generated: Correlation coefficients of the data, this is
        a dict of dataframes, containing keys ["means", "stds"] (e.g. output of
        compute_stats_over_dfs).

    :returns: Matplotlib figure and axes.
    """
    fig, axs = plt.subplots(2, 3, figsize=(10, 6), dpi=300)
    for i, (row, ax) in enumerate(
        zip(correlation_coefficients_data.index, axs.flatten())
    ):
        ax.set_title(row)
        ax.errorbar(
            range(3),
            correlation_coefficients_generated["means"].loc[row],
            label="Sample Ensemble",
            yerr=correlation_coefficients_generated["stds"].loc[row],
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
            c="tab:red",
            s=36,
            zorder=2,
            alpha=0.7,
        )
        ax.set_xlim((-0.75, 2.75))
        ax.set_xticks(ticks=range(3))
        ax.set_xticklabels(labels=correlation_coefficients_data.columns)
        ax.grid(alpha=0.7)
        if i == 0:
            ax.legend()

    plt.tight_layout()

    return fig, axs


def plot_qq(ax, data, generated, title, params, **kwargs):
    """
    Plots a QQ plot of the data against the generated on the provided ax.

    :param ax: Matplotlib axis.
    :param data: Array of data points.
    :param generated: Array of generated.
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
    ax.scatter(sorted(data), sorted(generated), **kwargs)
    ax.set_title(title)
    ax.set_xlim(params["xlims"])
    ax.set_ylim(params["ylims"])
    ax.set_xticks(params["xticks"])
    ax.set_yticks(params["yticks"])
    ax.set_xlabel("Data")
    ax.set_ylabel("Generated")
    ax.grid(alpha=0.7)

    return ax


def plot_qq_grid(data, generated, params):
    """
    Plots a 2x2 grid of QQ plots.

    :param data: Data must be a dataframe of shape (N, 4).
    :param generated: generated must be a dataframe with matching column names to the data.
    :param params: Additional parameter dictionary for ax configuration, required keys are
        ["xlims", "ylims", "xticks", "yticks"].

    :returns: Matplotlib figure and axes.
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=300, tight_layout=True)
    for column, ax in zip(data.columns, axs.flatten()):
        plot_qq(ax, data[column], generated[column], column, params)

    return fig, axs
