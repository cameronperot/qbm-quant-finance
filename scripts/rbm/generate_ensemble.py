import json
from joblib import Parallel, delayed
from pathlib import Path
from qbm.utils import get_project_dir, get_rng, load_artifact, unbinarize_df
from qbm.sampling import generate_samples_df

# configuration
project_dir = get_project_dir()

with open(project_dir / "scripts/rbm/config.json") as f:
    config = json.load(f)
model_name = config["model_name"]
ensemble_size = int(config["ensemble"]["size"])
n_steps = int(config["ensemble"]["n_steps"])
n_jobs = int(config["ensemble"]["n_jobs"])
seed = int(config["ensemble"]["seed"])

artifacts_dir = project_dir / f"artifacts/{model_name}"
save_dir = artifacts_dir / "ensemble_samples"
if not save_dir.exists():
    save_dir.mkdir()

# load the model and params
rng = get_rng(seed)
model = load_artifact(artifacts_dir / "model.pkl")
model_params = load_artifact(artifacts_dir / "params.pkl")

# generate initial values for the visible layer
V = rng.choice([0, 1], (ensemble_size, model_params["input_shape"][1]))


def generate_samples_df_wrapper(i):
    """
    Wrapper function to generate samples in parallel.

    :param i: Index value, must be within the range of the ensemble size.
    """
    # generate the samples
    samples, _ = generate_samples_df(
        model=model,
        v=V[i],
        n_samples=model_params["input_shape"][0],
        n_steps=n_steps,
        columns=model_params["columns"],
        binarization_params=model_params["binarization_params"],
        split_points=model_params["split_points"],
    )
    samples = unbinarize_df(samples, model_params["binarization_params"])

    # save the samples to csv
    samples.to_pickle(save_dir / f"{i+1:02}.pkl")


# run the jobs in parallel
Parallel(n_jobs=n_jobs)(
    delayed(generate_samples_df_wrapper)(i) for i in range(ensemble_size)
)
