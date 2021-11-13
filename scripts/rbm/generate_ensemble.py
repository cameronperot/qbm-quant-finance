import json
import numpy as np
import pandas as pd

from copy import deepcopy
from datetime import timedelta
from joblib import Parallel, delayed
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
model_id = config["model"]["id"]
ensemble_size = int(config["ensemble"]["size"])
n_steps = int(config["ensemble"]["n_steps"])
n_jobs = int(config["ensemble"]["n_jobs"])
seed = int(config["ensemble"]["seed"])

artifacts_dir = project_dir / f"artifacts/{model_id}"
save_dir = artifacts_dir / "samples_ensemble"
if not save_dir.exists():
    save_dir.mkdir()

# load the model and params
rng = get_rng(seed)
model = load_artifact(artifacts_dir / "model.pkl")
model_params = load_artifact(artifacts_dir / "model_params.json")

# generate initial values for the visible layer
V = rng.choice([0, 1], (ensemble_size, model_params["X_train_shape"][1]))

# load the transformer
transformer = None
if model_params["transform"].get("type") is not None:
    transformer = load_artifact(artifacts_dir / "transformer.pkl")


def generate_rbm_samples_df_wrapper(i):
    """
    Wrapper function to generate samples in parallel.

    :param i: Index value, must be within the range of the ensemble size.
    """
    iter_start_time = time()

    # copy the model and give it a new rng, otherwise it will always use the same random
    # number chain, which ends up as an attractor
    model_i = deepcopy(model)
    model_i.random_state = get_rng(i)

    # generate the samples
    samples, _ = generate_rbm_samples_df(
        model=model_i,
        v=V[i],
        n_samples=model_params["X_train_shape"][0],
        n_steps=n_steps,
        model_params=model_params,
    )
    samples = unbinarize_df(samples, model_params["binarization_params"])

    # transform the samples back to the original space
    if transformer is not None:
        if model_params["transform"]["type"] == "quantile":
            samples = pd.DataFrame(
                transformer.inverse_transform(samples),
                columns=samples.columns,
                index=samples.index,
            )
        elif model_params["transform"]["type"] == "power":
            samples = transformer.inverse_transform(samples)

    # save the samples to csv
    samples.to_pickle(save_dir / f"{i+1:03}.pkl")

    print(
        f"Completed iteration {i+1} of {ensemble_size} in {timedelta(seconds=time() - iter_start_time)}"
    )


# run the jobs in parallel
start_time = time()
Parallel(n_jobs=n_jobs)(
    delayed(generate_rbm_samples_df_wrapper)(i) for i in range(ensemble_size)
)

print(f"Completed {ensemble_size} iterations in {timedelta(seconds=time() - start_time)}")

# save config
save_artifact(config["ensemble"], save_dir / "ensemble_params.json")
