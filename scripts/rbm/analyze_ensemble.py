import itertools
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

from qbm.metrics import compute_annualized_volatility, compute_correlation_coefficients
from qbm.plotting import (
    plot_correlation_coefficients,
    plot_qq_grid,
    plot_tail_concentrations_grid,
    plot_volatility_comparison,
)
from qbm.utils import (
    get_project_dir,
    compute_stats_over_dfs,
    filter_df_on_values,
    kl_divergence,
    load_artifact,
    load_log_returns,
    save_artifact,
)

matplotlib.rcParams.update({"font.size": 14})


def main(model_id):
    # configuration
    project_dir = get_project_dir()

    if model_id is None:
        config = load_artifact(project_dir / "scripts/rbm/config.json")
        model_id = config["model"]["id"]

    artifacts_dir = project_dir / f"artifacts/{model_id}"
    data_dir = artifacts_dir / "samples_ensemble"
    results_dir = artifacts_dir / "results"
    if not results_dir.exists():
        results_dir.mkdir()
        (results_dir / "plots").mkdir()
        (results_dir / "data").mkdir()

    model_params = load_artifact(artifacts_dir / "config.json")["model"]

    # load the training data
    log_returns = pd.read_csv(
        artifacts_dir / "log_returns.csv", parse_dates=["date"], index_col="date"
    )

    # load the sampled ensemble data
    samples_ensemble_raw = [
        pd.read_pickle(data_dir / file_name)
        for file_name in sorted(data_dir.iterdir())
        if str(file_name).endswith(".pkl")
    ]

    # filter out binary indicator columns
    filter_columns = [
        column for column in samples_ensemble_raw[0].columns if column.endswith("_binary")
    ]
    samples_ensemble = [df.drop(filter_columns, axis=1) for df in samples_ensemble_raw]

    # compute KL divergences
    dkl_dfs = []
    for samples in samples_ensemble:
        dkl_dfs.append(
            pd.DataFrame.from_dict(
                {
                    column: kl_divergence(log_returns[column], samples[column], smooth=1e-6)
                    for column in log_returns.columns
                },
                orient="index",
            )
        )
    dkls = compute_stats_over_dfs(dkl_dfs)
    dkls = pd.DataFrame({k: v[0] for k, v in dkls.items()})
    dkls.to_csv(results_dir / "data/kl_divergences.csv", index_label="currency_pair")

    # compute the correlation coefficients
    combinations = list(itertools.combinations(log_returns.columns, 2))
    correlation_coefficients_data = compute_correlation_coefficients(
        log_returns, combinations
    )
    correlation_coefficients_sample = compute_stats_over_dfs(
        [
            compute_correlation_coefficients(samples, combinations)
            for samples in samples_ensemble
        ]
    )
    for k, v in correlation_coefficients_sample.items():
        correlation_coefficients_sample[k] = v.reindex_like(correlation_coefficients_data)

    correlation_coefficients_data.to_csv(
        results_dir / "data/correlation_coefficients_data.csv", index_label="currency_pairs"
    )
    correlation_coefficients_sample["means"].to_csv(
        results_dir / "data/correlation_coefficients_sample_means.csv",
        index_label="currency_pairs",
    )
    correlation_coefficients_sample["stds"].to_csv(
        results_dir / "data/correlation_coefficients_sample_stds.csv",
        index_label="currency_pairs",
    )
    correlation_coefficients_rmse = pd.DataFrame(
        np.sqrt(
            (
                (correlation_coefficients_data - correlation_coefficients_sample["means"])
                ** 2
            ).mean()
        ),
        columns=["RMSE"],
    )
    correlation_coefficients_rmse.to_csv(
        results_dir / "data/correlation_coefficients_rmse.csv",
        index_label="correlation_coefficient",
    )

    print("--------------------------------")
    print("Correlation Coefficients")
    print("--------------------------------")
    print("\tData")
    print(correlation_coefficients_data)
    print("\n\tSample (mean)")
    print(correlation_coefficients_sample["means"])
    print("\n\tSample (std)")
    print(correlation_coefficients_sample["stds"])
    print("\n\tRMSE")
    print(correlation_coefficients_rmse)
    print("--------------------------------\n\n")

    # compute the volatilities
    volatilities_data = compute_annualized_volatility(log_returns)
    volatilities_sample = compute_stats_over_dfs(
        [compute_annualized_volatility(samples) for samples in samples_ensemble]
    )
    volatilities = pd.DataFrame(
        {
            "Data": volatilities_data,
            "Sample Mean": volatilities_sample["means"],
            "Sample Std": volatilities_sample["stds"],
        }
    )
    volatilities.to_csv(results_dir / "data/volatilities.csv", index_label="currency_pair")

    print("--------------------------------")
    print("Annualized Volatility")
    print("--------------------------------")
    print(volatilities)
    print("--------------------------------\n\n")

    # compute the conditional volatilities
    if model_params["volatility_indicators"]:
        volatility_binarized = pd.read_csv(
            artifacts_dir / "volatility_binarized.csv",
            parse_dates=["date"],
            index_col="date",
        )
        log_returns_volatility = pd.merge(
            log_returns, volatility_binarized, left_index=True, right_index=True
        )
        currency_pairs = {
            column for column in model_params["columns"] if not column.endswith("_binary")
        }
        low_volatility_column_values = {
            f"{pair}_volatility_binary": 0 for pair in currency_pairs
        }
        high_volatility_column_values = {
            f"{pair}_volatility_binary": 1 for pair in currency_pairs
        }

        volatilities_low_data = compute_annualized_volatility(
            filter_df_on_values(log_returns_volatility, low_volatility_column_values)
        )
        volatilities_low_sample = compute_stats_over_dfs(
            [
                compute_annualized_volatility(
                    filter_df_on_values(samples, low_volatility_column_values)
                )
                for samples in samples_ensemble_raw
            ]
        )
        volatilities_low = pd.DataFrame(
            {
                "Data": volatilities_low_data,
                "Sample Mean": volatilities_low_sample["means"],
                "Sample Std": volatilities_low_sample["stds"],
            }
        )
        volatilities_low.to_csv(
            results_dir / "data/volatilities_low.csv", index_label="currency_pair"
        )

        volatilities_high_data = compute_annualized_volatility(
            filter_df_on_values(log_returns_volatility, high_volatility_column_values)
        )
        volatilities_high_sample = compute_stats_over_dfs(
            [
                compute_annualized_volatility(
                    filter_df_on_values(samples, high_volatility_column_values)
                )
                for samples in samples_ensemble_raw
            ]
        )
        volatilities_high = pd.DataFrame(
            {
                "Data": volatilities_high_data,
                "Sample Mean": volatilities_high_sample["means"],
                "Sample Std": volatilities_high_sample["stds"],
            }
        )
        volatilities_high.to_csv(
            results_dir / "data/volatilities_high.csv", index_label="currency_pair"
        )

        print("--------------------------------")
        print("Conditional Volatilities")
        print("--------------------------------")
        print("\tLow")
        print(volatilities_low)
        print("\n\tHigh")
        print(volatilities_high)
        print("--------------------------------\n\n")

    # compute the tails
    tails_data = pd.DataFrame(
        {
            "quantile_01": log_returns.quantile(0.01),
            "quantile_99": log_returns.quantile(0.99),
        }
    )
    quantiles_sample = []
    for samples in samples_ensemble:
        quantiles_sample.append(
            pd.DataFrame(
                {
                    "quantile_01": samples.quantile(0.01),
                    "quantile_99": samples.quantile(0.99),
                }
            )
        )
    tails_sample = compute_stats_over_dfs(quantiles_sample)

    tails_data.to_csv(results_dir / "data/tails_data.csv", index_label="currency_pair")
    tails_sample["means"].to_csv(
        results_dir / "data/tails_sample_means.csv", index_label="currency_pair"
    )
    tails_sample["stds"].to_csv(
        results_dir / "data/tails_sample_stds.csv", index_label="currency_pair"
    )

    print("--------------------------------")
    print("Tails")
    print("--------------------------------")
    print("\tData")
    print(tails_data)
    print("\n\tSample (mean)")
    print(tails_sample["means"])
    print("\n\tSample (std)")
    print(tails_sample["stds"])
    print("--------------------------------\n\n")

    # plot the correlation coefficients
    fig, axs = plot_correlation_coefficients(
        correlation_coefficients_data, correlation_coefficients_sample
    )
    plt.savefig(results_dir / "plots/correlation_coefficients.png")
    plt.close(fig)

    # plot the volatilities
    volatility_plot_params = {
        "xlims": (-0.75, 3.75),
        "ylims": (0.08, 0.11),
        "yticks": np.linspace(0.08, 0.11, 7),
    }
    fig, ax = plot_volatility_comparison(
        volatilities_data, volatilities_sample, volatility_plot_params
    )
    plt.savefig(results_dir / "plots/volatilities.png")
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze the statistical ensemble.")
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="model_id of the form BernoulliRBM_YYYYMMDD_HHMMSS",
    )
    args = parser.parse_args()

    main(args.model_id)
