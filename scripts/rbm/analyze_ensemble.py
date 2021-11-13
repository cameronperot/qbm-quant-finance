import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

from qbm.metrics import compute_annualized_volatility, compute_correlation_coefficients
from qbm.plotting import (
    plot_correlation_coefficients,
    plot_qq_grid,
    plot_volatility_comparison,
)
from qbm.utils import (
    get_project_dir,
    compute_stats_over_dfs,
    load_artifact,
    load_log_returns,
)

# configuration
project_dir = get_project_dir()

config = load_artifact(project_dir / "scripts/rbm/config.json")
model_id = config["model"]["id"]

artifacts_dir = project_dir / f"artifacts/{model_id}"
data_dir = artifacts_dir / "samples_ensemble"
plot_dir = artifacts_dir / "plots"
if not plot_dir.exists():
    plot_dir.mkdir()

# load the training data
log_returns = load_log_returns(artifacts_dir / "log_returns.csv")

# load the sampled ensemble data
samples_ensemble = [
    pd.read_pickle(data_dir / file_name)
    for file_name in data_dir.iterdir()
    if str(file_name).endswith(".pkl")
]

# filter out binary indicator columns
filter_columns = [
    column for column in samples_ensemble[0].columns if column.endswith("_binary")
]
samples_ensemble = [df.drop(filter_columns, axis=1) for df in samples_ensemble]

# compute QQ distance
qq_rmse = {column: [] for column in log_returns.columns}
qq_rmse = np.zeros((len(samples_ensemble), len(log_returns.columns)))
for i, samples in enumerate(samples_ensemble):
    for j, column in enumerate(log_returns.columns):
        qq_rmse[i, j] = mean_squared_error(
            np.sort(samples[column]), np.sort(log_returns[column]), squared=False
        )

qq_rmse_means = qq_rmse.mean(axis=0)
qq_rmse_stds = qq_rmse.std(axis=0)
qq_extrema = {
    "min": np.argmin(qq_rmse.mean(axis=1)),
    "max": np.argmax(qq_rmse.mean(axis=1)),
}

qq_rmse = {
    column: {"mean": qq_rmse_means[j], "std": qq_rmse_stds[j]}
    for j, column in enumerate(log_returns.columns)
}
qq_rmse = pd.DataFrame.from_dict(qq_rmse, orient="index")
qq_rmse.to_csv(data_dir / "qq_rmse.csv")

print("--------------------------------")
print("QQ RMSE")
print("--------------------------------")
print(qq_rmse)
print("--------------------------------\n\n")

# compute the correlation coefficients
combinations = (
    ("EURUSD", "GBPUSD"),
    ("EURUSD", "USDJPY"),
    ("EURUSD", "USDCAD"),
    ("GBPUSD", "USDJPY"),
    ("GBPUSD", "USDCAD"),
    ("USDJPY", "USDCAD"),
)
correlation_coefficients_data = compute_correlation_coefficients(log_returns, combinations)
correlation_coefficients_sample = compute_stats_over_dfs(
    [
        compute_correlation_coefficients(samples, combinations)
        for samples in samples_ensemble
    ]
)
for k, v in correlation_coefficients_sample.items():
    correlation_coefficients_sample[k] = v.reindex_like(correlation_coefficients_data)

correlation_coefficients_data.to_csv(data_dir / "correlation_coefficients_data.csv")
correlation_coefficients_sample["means"].to_csv(
    data_dir / "correlation_coefficients_sample_means.csv"
)
correlation_coefficients_sample["stds"].to_csv(
    data_dir / "correlation_coefficients_sample_stds.csv"
)
correlation_coefficients_rmse = pd.DataFrame(
    np.sqrt(
        (
            (correlation_coefficients_data - correlation_coefficients_sample["means"]) ** 2
        ).mean()
    ),
    columns=["RMSE"],
)
correlation_coefficients_rmse.to_csv(data_dir / "correlation_coefficients_rmse.csv")

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
volatilities.to_csv(data_dir / "volatilities.csv")

print("--------------------------------")
print("Annualized Volatility")
print("--------------------------------")
print(volatilities)
print("--------------------------------\n\n")

# compute the tails
tails_data = pd.DataFrame(
    {"quantile_01": log_returns.quantile(0.01), "quantile_99": log_returns.quantile(0.99),}
)
quantiles_sample = []
for samples in samples_ensemble:
    quantiles_sample.append(
        pd.DataFrame(
            {"quantile_01": samples.quantile(0.01), "quantile_99": samples.quantile(0.99),}
        )
    )
tails_sample = compute_stats_over_dfs(quantiles_sample)

tails_data.to_csv(data_dir / "tails_data.csv")
tails_sample["means"].to_csv(data_dir / "tails_sample_means.csv")
tails_sample["stds"].to_csv(data_dir / "tails_sample_stds.csv")

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

# QQ plots
qq_plot_params = {
    "title": "test",
    "xlims": (-0.045, 0.045),
    "ylims": (-0.045, 0.045),
    "xticks": np.linspace(-0.04, 0.04, 9),
    "yticks": np.linspace(-0.04, 0.04, 9),
}
for extrema, i in qq_extrema.items():
    fig, axs = plot_qq_grid(log_returns, samples_ensemble[i], qq_plot_params)
    plt.savefig(plot_dir / f"qq_{extrema}.png")
    plt.close(fig)

# plot the correlation coefficients
fig, axs = plot_correlation_coefficients(
    correlation_coefficients_data, correlation_coefficients_sample
)
plt.savefig(plot_dir / "correlation_coefficients.png")
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
plt.savefig(plot_dir / "volatilities.png")
plt.close(fig)
