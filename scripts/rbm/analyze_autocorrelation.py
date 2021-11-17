import emcee
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm  # use statsmodels' acf because it's faster and smoother than emcee's

from qbm.plotting import plot_autocorrelation_grid
from qbm.utils import get_project_dir, load_artifact, load_log_returns


def main(model_id):
    # configuration
    project_dir = get_project_dir()

    config = None
    if model_id is None:
        config = load_artifact(project_dir / "scripts/rbm/config.json")
        model_id = config["model"]["id"]

    artifacts_dir = project_dir / f"artifacts/{model_id}"
    results_dir = artifacts_dir / "results"
    data_dir = artifacts_dir / "autocorrelation_samples"
    if not results_dir.exists():
        results_dir.mkdir()
        (results_dir / "plots").mkdir()
        (results_dir / "data").mkdir()

    if config is None:
        config = load_artifact(artifacts_dir / "config.json")

    n_lags = int(config["autocorrelation"]["n_lags"])

    # load the raw data
    log_returns = pd.read_csv(
        artifacts_dir / "log_returns.csv", parse_dates=["date"], index_col="date"
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
    print("--------------------------------")
    print("Integrated Autocorrelation Times")
    print("--------------------------------")
    for column in samples.columns:
        if column.endswith("_binary"):
            continue
        acfs[column] = sm.tsa.stattools.acf(samples[column], nlags=n_lags, fft=True)
        integrated_times[column] = {
            "autocorrelation_time": emcee.autocorr.integrated_time(samples[column])[0]
        }
    integrated_times = pd.DataFrame.from_dict(integrated_times, orient="index")
    print(integrated_times)
    print("--------------------------------")
    acfs = pd.DataFrame(acfs)
    acfs.index.rename("lag", inplace=True)
    acfs.to_csv(results_dir / "data/autocorrelation_functions.csv", index_label="lag")
    integrated_times.to_csv(
        results_dir / "data/autocorrelation_times.csv", index_label="currency_pair"
    )

    # plot the acfs
    fig, ax = plot_autocorrelation_grid(acfs)
    plt.savefig(results_dir / "plots/autocorrelations.png")
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze the autocorrelations.")
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="model_id of the form BernoulliRBM_YYYYMMDD_HHMMSS",
    )
    args = parser.parse_args()

    main(args.model_id)
