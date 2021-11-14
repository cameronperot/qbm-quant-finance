import json
import pandas as pd

from qbm.utils import get_project_dir, load_artifact

pd.set_option("display.max_columns", None)
# pd.set_option("display.max_colwidth", 200)
project_dir = get_project_dir()
artifacts_dir = project_dir / "artifacts"
artifacts_subdirs = sorted(
    [
        x
        for x in artifacts_dir.iterdir()
        if x.is_dir() and str(x.stem).startswith("BernoulliRBM_")
    ]
)

rows = {}
qq_rmse = {}
cc_rmse = {}
for dir in artifacts_subdirs:
    config = load_artifact(dir / "config.json")
    model_params = config["model"]
    ensemble_params = config.get("ensemble", {})
    autocorrelation_params = config.get("autocorrelation", {})

    rows[dir.stem] = {
        "seed": model_params["seed"],
        "n_bits": model_params["n_bits"],
        "n_components": model_params["n_components"],
        "learning_rate": model_params["learning_rate"],
        "n_iter": model_params["n_iter"],
        "lr_decay_epoch": model_params["lr_decay_epoch"],
        "lr_decay_period": model_params["lr_decay_period"],
        "batch_size": model_params["batch_size"],
        "volatility_indicators": model_params["volatility_indicators"],
        "transform": model_params["transform"],
    }

    rows[dir.stem]["ensemble_size"] = ensemble_params.get("size")
    rows[dir.stem]["ensemble_n_steps"] = ensemble_params.get("n_steps")

    try:
        qq_rmse[model_params["id"]] = pd.read_csv(
            dir / "samples_ensemble/qq_rmse.csv", index_col=0
        )["mean"]
        cc_rmse[model_params["id"]] = pd.read_csv(
            dir / "samples_ensemble/correlation_coefficients_rmse.csv", index_col=0
        )["RMSE"]
    except:
        pass


rows = pd.DataFrame.from_dict(rows, orient="index").sort_index()
qq_rmse = pd.DataFrame(qq_rmse)
cc_rmse = pd.DataFrame(cc_rmse)

print("Artifacts")
print(rows)
print("\nQQ RMSE")
print(qq_rmse)
print("\nQQ RMSE Mean")
print(qq_rmse.mean())
print("\nCorrelation Coefficients RMSE")
print(cc_rmse)
print("\nCorrelation Coefficients Mean")
print(cc_rmse.mean())
