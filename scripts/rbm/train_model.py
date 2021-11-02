import numpy as np

from datetime import datetime
from sklearn.neural_network import BernoulliRBM

from qbm.utils import (
    binarize_df,
    convert_binarized_df_to_input_array,
    get_project_dir,
    get_rng,
    load_artifact,
    load_train_data,
    lr_exp_decay,
    save_artifact,
)

# configuration
project_dir = get_project_dir()

config = load_artifact(project_dir / "scripts/rbm/config.json")
model_params = config["model_params"]
model_params["name"] = f"BernoulliRBM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

artifacts_dir = project_dir / f"artifacts/{model_params['name']}"
data_dir = project_dir / "data"

rng = get_rng(model_params["seed"])

# data loading
log_returns = load_train_data(data_dir)

# binarization
binarization_params = {}
for column in log_returns.columns:
    binarization_params[column] = {
        "n_bits": config["model_params"]["n_bits"],
        "x_min": log_returns[column].min(),
        "x_max": log_returns[column].max(),
    }
log_returns_binarized = binarize_df(log_returns, binarization_params)
model_params["binarization_params"] = binarization_params

# create the training set
X_train = convert_binarized_df_to_input_array(log_returns_binarized)
for i in range(X_train.shape[0]):
    assert "".join([x for x in log_returns_binarized.iloc[i]]) == "".join(
        [str(x) for x in X_train[i]]
    )
rng.shuffle(X_train)
model_params["input_shape"] = X_train.shape
model_params["columns"] = log_returns.columns.to_list()

# train model
print(f"Model Name: {model_params['name']}")
model = BernoulliRBM(
    n_components=model_params["n_components"],
    learning_rate=model_params["learning_rate"],
    learning_rate_schedule=lr_exp_decay(
        np.arange(model_params["n_iter"]),
        model_params["lr_decay_epoch"],
        model_params["lr_decay_period"],
    ),
    early_stopping_criteria=model_params["early_stopping_criteria"],
    batch_size=model_params["batch_size"],
    n_iter=model_params["n_iter"],
    random_state=rng,
    verbose=100,
)
model.fit(X_train)

# save artifacts
save_artifact(model_params, artifacts_dir / "params.json")
save_artifact(model, artifacts_dir / "model.pkl")
log_returns.to_csv(artifacts_dir / "log_returns.csv")
