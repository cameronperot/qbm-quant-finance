import numpy as np
import pandas as pd
import pytest

from qbm.utils import (
    binarize_df,
    convert_bin_str_to_list,
    get_binarization_params,
    prepare_training_data,
)


@pytest.fixture
def training_input(request):
    n_samples, n_bits, offset = request.param
    log_returns = pd.DataFrame(
        {
            "a": np.arange(n_samples),
            "b": np.linspace(0, 10, n_samples),
            "c": np.linspace(-10, 10, n_samples),
            "d": np.linspace(10, -10, n_samples),
        }
    )

    binarization_params = get_binarization_params(log_returns, n_bits)
    log_returns_binarized = binarize_df(log_returns, binarization_params)

    n_samples_additional = n_samples - offset
    additional_variables = pd.DataFrame(
        {
            "a_binary": np.zeros(n_samples_additional),
            "b_binary": np.ones(n_samples_additional),
        }
    ).astype(np.int8)
    additional_variables.index = range(offset, n_samples)

    return log_returns_binarized, additional_variables, request.param


@pytest.mark.parametrize(
    "training_input", [(100, 16, 10), (99, 15, 9)], indirect=True,
)
def test_prepare_training_data_without_additional_variables(training_input):
    log_returns_binarized, additional_variables, param = training_input
    n_samples, n_bits, offset = param

    training_data = prepare_training_data(log_returns_binarized)

    for i, column in enumerate(log_returns_binarized.columns):
        x = np.stack(log_returns_binarized[column].map(convert_bin_str_to_list))
        x_train = training_data["X_train"][:, i * n_bits : (i + 1) * n_bits]
        assert (x == x_train).all()

    assert (training_data["columns"] == log_returns_binarized.columns).all()
    assert training_data["split_indices"] == [
        i for i in range(n_bits, log_returns_binarized.shape[1] * n_bits, n_bits)
    ]
    assert (training_data["index"] == log_returns_binarized.index).all()


@pytest.mark.parametrize(
    "training_input", [(100, 16, 10), (99, 15, 9)], indirect=True,
)
def test_prepare_training_data_with_additional_variables(training_input):
    log_returns_binarized, additional_variables, param = training_input
    n_samples, n_bits, offset = param
    n_columns_log_returns = log_returns_binarized.shape[1]
    n_columns_additional = additional_variables.shape[1]

    training_data = prepare_training_data(log_returns_binarized, additional_variables)

    for i, column in enumerate(log_returns_binarized.columns):
        x = np.stack(log_returns_binarized[column].map(convert_bin_str_to_list))[offset:]
        x_train = training_data["X_train"][:, i * n_bits : (i + 1) * n_bits]
        assert (x == x_train).all()

    for i, column in enumerate(additional_variables.columns):
        i += n_columns_log_returns * n_bits
        x = additional_variables[column]
        x_train = training_data["X_train"][:, i].flatten()
        assert (x == x_train).all()

    assert (
        training_data["columns"]
        == log_returns_binarized.columns.to_list() + additional_variables.columns.to_list()
    )
    assert training_data["split_indices"] == [
        i for i in range(n_bits, n_columns_log_returns * n_bits, n_bits)
    ] + [
        i
        for i in range(
            n_bits * n_columns_log_returns,
            n_bits * n_columns_log_returns + n_columns_additional,
        )
    ]
    assert (training_data["index"] == additional_variables.index).all()


@pytest.mark.parametrize(
    "training_input", [(100, 16, 10), (99, 15, 9)], indirect=True,
)
def test_prepare_training_data_with_invalid_additional_variables(training_input):
    log_returns_binarized, additional_variables, param = training_input
    n_samples, n_bits, offset = param
    additional_variables["invalid"] = np.arange(additional_variables.shape[0])

    with pytest.raises(Exception):
        prepare_training_data(log_returns_binarized, additional_variables)
