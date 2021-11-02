import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qbm.metrics import compute_annualized_volatility, compute_correlation_coefficients
from qbm.plotting import plot_correlation_coefficients, plot_qq_grid, plot_volatilities
from qbm.utils import get_project_dir, compute_stats_over_dfs, load_artifact

# configuration
project_dir = get_project_dir()

config = load_artifact(project_dir / "scripts/rbm/config.json")
model_name = config["load_model_name"]

artifacts_dir = project_dir / f"artifacts/{model_name}"
data_dir = artifacts_dir / "ensemble_samples"
plot_dir = artifacts_dir / "plots"
if not plot_dir.exists():
    plot_dir.mkdir()

# load the raw data
log_returns = pd.read_csv(
    artifacts_dir / "log_returns.csv",
    parse_dates=["date"],
    date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d"),
    index_col="date",
)

# load the ensemble data
sample_ensemble = [
    pd.read_pickle(data_dir / file_name)
    for file_name in data_dir.iterdir()
    if str(file_name).endswith(".pkl")
]

# QQ plots
qq_plot_params = {
    "title": "test",
    "xlims": (-0.045, 0.045),
    "ylims": (-0.045, 0.045),
    "xticks": np.linspace(-0.04, 0.04, 9),
    "yticks": np.linspace(-0.04, 0.04, 9),
}
for i, samples in enumerate(sample_ensemble):
    fig, axs = plot_qq_grid(log_returns, samples, qq_plot_params)
    plt.savefig(plot_dir / f"qq_{i+1:03}.png")
    plt.close(fig)

# compute the correlation coefficients
combinations = (
    ("EURUSD", "GBPUSD"),
    ("EURUSD", "USDJPY"),
    ("EURUSD", "USDCAD"),
    ("GBPUSD", "USDJPY"),
    ("GBPUSD", "USDCAD"),
    ("USDJPY", "USDCAD"),
)

correlation_coefficients_data = compute_correlation_coefficients(
    log_returns, combinations
)
correlation_coefficients_sample = compute_stats_over_dfs(
    [
        compute_correlation_coefficients(samples, combinations)
        for samples in sample_ensemble
    ]
)
for k, v in correlation_coefficients_sample.items():
    correlation_coefficients_sample[k] = v.reindex_like(correlation_coefficients_data)

print("Data")
print(correlation_coefficients_data)
print("\nSample (mean)")
print(correlation_coefficients_sample["means"])
print("\nSample (std)")
print(correlation_coefficients_sample["stds"])

fig, axs = plot_correlation_coefficients(
    correlation_coefficients_data, correlation_coefficients_sample
)
plt.savefig(plot_dir / "correlation_coefficients.png")
plt.close(fig)

# compute the volatilities
volatilities_data = compute_annualized_volatility(log_returns)
volatilities_sample = compute_stats_over_dfs(
    [compute_annualized_volatility(samples) for samples in sample_ensemble]
)
volatilities = pd.DataFrame(
    {
        "Data": volatilities_data,
        "Sample Mean": volatilities_sample["means"],
        "Sample Std": volatilities_sample["stds"],
    }
)
print("\nAnnualized Volatility")
print(volatilities)

# plot the volatilities
volatility_plot_params = {
    "xlims": (-0.75, 3.75),
    "ylims": (0.08, 0.11),
    "yticks": np.linspace(0.08, 0.11, 7),
}
fig, ax = plot_volatilities(
    volatilities_data, volatilities_sample, volatility_plot_params
)
plt.savefig(plot_dir / "volatilities.png")
plt.close(fig)
