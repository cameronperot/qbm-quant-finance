import pandas as pd

from datetime import timedelta
from time import time

from qbm.utils import (
    get_project_dir,
    get_rng,
    load_artifact,
    save_artifact,
    unbinarize_df,
)
from qbm.sampling import generate_rbm_samples_df

# configuration
project_dir = get_project_dir()

config = load_artifact(project_dir / "scripts/rbm/config.json")
autocorrelation_params = config["autocorrelation"]
model_id = config["model"]["id"]
n_samples_per_df = int(autocorrelation_params["n_samples_per_df"])
n_sample_dfs = int(autocorrelation_params["n_sample_dfs"])

artifacts_dir = project_dir / f"artifacts/{model_id}"
save_dir = artifacts_dir / "autocorrelation_samples"
if not save_dir.exists():
    save_dir.mkdir()

# load model and params
rng = get_rng(42)
model = load_artifact(artifacts_dir / "model.pkl")
model_params = load_artifact(artifacts_dir / "config.json")["model"]

# generate initial values for the visible layer
v = rng.choice([0, 1], model_params["X_train_shape"][1])

# load the transformer
transformer = None
if model_params["transform"].get("type") is not None:
    transformer = load_artifact(artifacts_dir / "transformer.pkl")


# generate and save the samples
start_time = time()
for i in range(n_sample_dfs):
    iter_start_time = time()

    # generate the samples
    autocorrelation_samples, v = generate_rbm_samples_df(
        model=model, v=v, n_samples=n_samples_per_df, n_steps=1, model_params=model_params,
    )
    autocorrelation_samples = unbinarize_df(
        autocorrelation_samples, model_params["binarization_params"]
    )

    # transform the samples back to the original space
    if transformer is not None:
        if model_params["transform"]["type"] == "quantile":
            autocorrelation_samples = pd.DataFrame(
                transformer.inverse_transform(autocorrelation_samples),
                columns=autocorrelation_samples.columns,
                index=autocorrelation_samples.index,
            )
        elif model_params["transform"]["type"] == "power":
            autocorrelation_samples = transformer.inverse_transform(autocorrelation_samples)

    # increment the index (useful for when recombining the csv files)
    autocorrelation_samples.index += i * n_samples_per_df

    # save the samples to csv
    autocorrelation_samples.to_pickle(save_dir / f"{i+1:03}.pkl")

    print(
        f"Completed iteration {i+1} of {n_sample_dfs} in {timedelta(seconds=time() - iter_start_time)}"
    )

print(f"Completed {n_sample_dfs} iterations in {timedelta(seconds=time() - start_time)}")

# update the config for this run
config = load_artifact(artifacts_dir / "config.json")
config["autocorrelation"] = autocorrelation_params
save_artifact(config, artifacts_dir / "config.json")
