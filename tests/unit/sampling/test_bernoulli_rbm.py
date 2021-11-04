import numpy as np
import pytest

from unittest.mock import MagicMock

from qbm.sampling import generate_rbm_sample, generate_rbm_samples_df


@pytest.mark.parametrize("n_visible, n_steps", [(1, 1), (10, 10)])
def test_generate_rbm_sample(n_visible, n_steps):
    mock_model = MagicMock()
    mock_model.gibbs.return_value = np.ones(n_visible)
    v = np.zeros(n_visible)

    v = generate_rbm_sample(mock_model, v, n_steps)

    assert (v == np.ones(n_visible)).all()
    assert v.dtype == np.int8


@pytest.mark.parametrize(
    "params",
    [
        {
            "n_visible": 34,
            "n_samples": 1,
            "n_steps": 1,
            "n_bits": 16,
            "columns": ["a", "b", "a_binary", "b_binary"],
            "split_indices": [16, 32, 33],
        },
        {
            "n_visible": 34,
            "n_samples": 10,
            "n_steps": 10,
            "n_bits": 16,
            "columns": ["a", "b", "a_binary", "b_binary"],
            "split_indices": [16, 32, 33],
        },
        {
            "n_visible": 33,
            "n_samples": 10,
            "n_steps": 10,
            "n_bits": 16,
            "columns": ["a", "b", "a_binary"],
            "split_indices": [16, 32],
        },
        {
            "n_visible": 32,
            "n_samples": 10,
            "n_steps": 10,
            "n_bits": 15,
            "columns": ["a", "b", "a_binary", "b_binary"],
            "split_indices": [15, 30, 31],
        },
    ],
)
def test_generate_rbm_samples_df_sequential(params):
    mock_model = MagicMock()
    mock_model.gibbs.return_value = np.ones(params["n_visible"])
    v = np.zeros(params["n_visible"])

    df, v = generate_rbm_samples_df(
        model=mock_model,
        v=v,
        n_samples=params["n_samples"],
        n_steps=params["n_steps"],
        model_params=params,
    )

    for column in params["columns"]:
        if column.endswith("_binary"):
            assert set(df[column]) <= set(["0", "1"])
        else:
            assert (df[column].map(len) == params["n_bits"]).all()
            assert (df[column] == "1" * params["n_bits"]).all()

    df = df.applymap(lambda x: int(x, 2))
    for column in params["columns"]:
        if column.endswith("_binary"):
            assert set(df[column]) <= set([0, 1])
        else:
            assert (df[column] == 2 ** params["n_bits"] - 1).all()

    assert mock_model.gibbs.call_count == params["n_samples"] * params["n_steps"]


@pytest.mark.parametrize(
    "params",
    [
        {
            "n_visible": 34,
            "n_samples": 1,
            "n_steps": 1,
            "n_bits": 16,
            "columns": ["a", "b", "a_binary", "b_binary"],
            "split_indices": [16, 32, 33],
        },
        {
            "n_visible": 34,
            "n_samples": 10,
            "n_steps": 10,
            "n_bits": 16,
            "columns": ["a", "b", "a_binary", "b_binary"],
            "split_indices": [16, 32, 33],
        },
        {
            "n_visible": 33,
            "n_samples": 10,
            "n_steps": 10,
            "n_bits": 16,
            "columns": ["a", "b", "a_binary"],
            "split_indices": [16, 32],
        },
        {
            "n_visible": 32,
            "n_samples": 10,
            "n_steps": 10,
            "n_bits": 15,
            "columns": ["a", "b", "a_binary", "b_binary"],
            "split_indices": [15, 30, 31],
        },
    ],
)
def test_generate_rbm_samples_df_parallel(params):
    mock_model = MagicMock()
    mock_model.gibbs.return_value = np.ones((params["n_samples"], params["n_visible"]))
    v = np.zeros((params["n_samples"], params["n_visible"]))

    df, v = generate_rbm_samples_df(
        model=mock_model,
        v=v,
        n_samples=params["n_samples"],
        n_steps=params["n_steps"],
        model_params=params,
    )

    for column in params["columns"]:
        if column.endswith("_binary"):
            assert set(df[column]) <= set(["0", "1"])
        else:
            assert (df[column].map(len) == params["n_bits"]).all()
            assert (df[column] == "1" * params["n_bits"]).all()

    df = df.applymap(lambda x: int(x, 2))
    for column in params["columns"]:
        if column.endswith("_binary"):
            assert set(df[column]) <= set([0, 1])
        else:
            assert (df[column] == 2 ** params["n_bits"] - 1).all()

    assert mock_model.gibbs.call_count == params["n_steps"]
