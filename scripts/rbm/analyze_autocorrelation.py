import emcee
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm  # use statsmodels' acf because it's faster and smoother than emcee's

from qbm.plotting import plot_autocorrelation_grid
from qbm.utils import get_project_dir

# configuration
project_dir = get_project_dir()

with open(project_dir / "scripts/rbm/config.json") as f:
    config = json.load(f)
model_name = config["model_name"]
n_lags = int(config["autocorrelation"]["n_lags"])

artifacts_dir = project_dir / f"artifacts/{model_name}"
data_dir = artifacts_dir / "autocorrelation_samples"
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

# load the sample chain
file_names = [
    file_name for file_name in data_dir.iterdir() if str(file_name).endswith("pkl")
]
samples = pd.concat([pd.read_pickle(data_dir / file_name) for file_name in file_names])

# compute the integrated autocorrelation times and acfs
lags = range(n_lags + 1)
integrated_times = {}
acfs = {}
for column in samples.columns:
    acfs[column] = sm.tsa.stattools.acf(samples[column], nlags=n_lags, fft=True)
    integrated_times[column] = emcee.autocorr.integrated_time(samples[column])[0]
    print(f"{column} integrated autocorrelation time = {integrated_times[column]:.2f}")
acfs = pd.DataFrame(acfs)

# plot the acfs
fig, ax = plot_autocorrelation_grid(acfs)
plt.savefig(plot_dir / "autocorrelations.png")
plt.close(fig)
