import json

from datetime import timedelta
from time import time

from qbm.utils import get_project_dir, get_rng, load_artifact, unbinarize_df
from qbm.sampling import generate_samples_df

# configuration
project_dir = get_project_dir()

with open(project_dir / "scripts/rbm/config.json") as f:
    config = json.load(f)
model_name = config["model_name"]
n_samples_per_df = int(config["autocorrelation"]["n_samples_per_df"])
n_sample_dfs = int(config["autocorrelation"]["n_sample_dfs"])

artifacts_dir = project_dir / f"artifacts/{model_name}"
save_dir = artifacts_dir / "autocorrelation_samples"
if not save_dir.exists():
    save_dir.mkdir()

# load model and params
rng = get_rng(42)
model = load_artifact(artifacts_dir / "model.pkl")
model_params = load_artifact(artifacts_dir / "params.pkl")

# generate initial values for the visible layer
v = rng.choice([0, 1], model_params["input_shape"][1])

# generate and save the samples
start_time = time()
for i in range(n_sample_dfs):
    iter_start_time = time()

    # generate the samples
    autocorrelation_samples, v = generate_samples_df(
        model=model,
        v=v,
        n_samples=n_samples_per_df,
        n_steps=1,
        columns=model_params["columns"],
        binarization_params=model_params["binarization_params"],
        split_points=model_params["split_points"],
    )
    autocorrelation_samples = unbinarize_df(
        autocorrelation_samples, model_params["binarization_params"]
    )

    # increment the index (useful for when recombining the csv files)
    autocorrelation_samples.index += i * n_samples_per_df

    # save the samples to csv
    autocorrelation_samples.to_pickle(save_dir / f"{i+1:03}.pkl")

    print(
        f"Completed iteration {i+1} of {n_sample_dfs} in {timedelta(seconds=time() - iter_start_time)}"
    )

print(
    f"Completed {n_sample_dfs} iterations in {timedelta(seconds=time() - start_time)}"
)
