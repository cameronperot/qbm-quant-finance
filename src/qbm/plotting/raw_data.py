import matplotlib.pyplot as plt


def plot_histogram_grid(df, params, **kwargs):
    """
    Plots a grid of historgrams for the desired feature.

    :param df: Dataframe.
    :param params: Additional parameter dictionary for ax configuration, required keys are
        ["xlims", "ylims", "xticks", "yticks"].

    :returns: Matplotlib figure and axes.
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), tight_layout=True, dpi=300)
    for column, ax in zip(df.columns, axs.flatten()):
        ax.hist(df[column], **kwargs)
        ax.set_title(pair)
        ax.set_xlim(params["xlims"])
        ax.set_ylim(params["ylims"])
        ax.set_xticks(params["xticks"])
        ax.set_yticks(params["yticks"])
        ax.grid(alpha=0.7)

    plt.tight_layout()

    return fig, axs


def plot_violin(df, params, **kwargs):
    """
    Plots a violin plot with a box plot overlayed.

    :param df: Dataframe.
    :param params: Additional parameter dictionary for ax configuration, required keys are
        ["ylims", "yticks"].

    :returns: Matplotlib figure and axis.
    """
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.violinplot(df, showextrema=False)
    ax.boxplot(
        df,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "none", "markeredgecolor": "b",},
        medianprops={"color": "g"},
        flierprops={"marker": "x", "markeredgecolor": "tab:red"},
    )
    ax.set_xticks(range(1, len(df.columns) + 1))
    ax.set_xticklabels(df.columns)
    ax.set_ylim(params["ylims"])
    ax.set_yticks(params["yticks"])
    ax.grid(alpha=0.7)

    plt.tight_layout()

    return fig, ax
