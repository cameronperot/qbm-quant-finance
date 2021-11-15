import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import QuantileTransformer

from qbm.metrics import compute_rolling_volatility
from qbm.utils import (
    binarize_df,
    binarize_volatility,
    get_binarization_params,
    get_project_dir,
    get_rng,
    load_artifact,
    load_log_returns,
    lr_exp_decay,
    prepare_training_data,
    save_artifact,
    PowerTransformer,
)

# configuration
project_dir = get_project_dir()

config = load_artifact(project_dir / "scripts/rbm/config.json")
model_params = config["model"].copy()
data_params = config["data"]
model_params["id"] = f"BernoulliRBM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
config["model"]["id"] = model_params["id"]

artifacts_dir = project_dir / f"artifacts/{model_params['id']}"
data_dir = project_dir / "data"

rng = get_rng(model_params["seed"])

# data loading
date_format = "%Y-%m-%d"
start_date = datetime.strptime(data_params["start_date"], date_format)
end_date = datetime.strptime(data_params["end_date"], date_format)
if model_params["volatility_indicators"]:
    start_date -= timedelta(days=90)

log_returns = load_log_returns(
    data_params["data_source"],
    start_date=start_date,
    end_date=end_date,
    outlier_threshold=data_params["outlier_threshold"],
)
log_returns_raw = log_returns.copy()

# data transformation
transformer = None
if model_params["transform"].get("type") is not None:
    if model_params["transform"]["type"] == "quantile":
        transformer = QuantileTransformer(**model_params["transform"]["params"])
        log_returns = pd.DataFrame(
            transformer.fit_transform(log_returns),
            columns=log_returns.columns,
            index=log_returns.index,
        )
    elif model_params["transform"]["type"] == "power":
        transformer = PowerTransformer(log_returns, **model_params["transform"]["params"])
        log_returns = transformer.transform(log_returns)


# binarization
binarization_params = get_binarization_params(log_returns, n_bits=16)
log_returns_binarized = binarize_df(log_returns, binarization_params)
model_params["binarization_params"] = binarization_params

# volatility indicators
volatility_binarized = None
if model_params["volatility_indicators"]:
    volatility_binarized = binarize_volatility(
        compute_rolling_volatility(log_returns, timedelta(days=90))
    )

# create the training set
training_data = prepare_training_data(log_returns_binarized, volatility_binarized)
X_train = training_data["X_train"]
rng.shuffle(X_train)
model_params["X_train_shape"] = X_train.shape
model_params["columns"] = training_data["columns"]
model_params["split_indices"] = training_data["split_indices"]

# train model
print(f"Model Name: {model_params['id']}")
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
save_artifact(config, project_dir / "scripts/rbm/config.json")
save_artifact({"model": model_params, "data": data_params}, artifacts_dir / "config.json")
save_artifact(model, artifacts_dir / "model.pkl")
log_returns_raw.loc[training_data["index"]].to_csv(artifacts_dir / "log_returns.csv")
if volatility_binarized is not None:
    volatility_binarized.loc[training_data["index"]].to_csv(
        artifacts_dir / "volatility_binarized.csv"
    )
if transformer is not None:
    save_artifact(transformer, artifacts_dir / "transformer.pkl")
