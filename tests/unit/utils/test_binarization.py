import numpy as np
import pandas as pd
import pytest

from qbm.utils import (
    binarize,
    binarize_df,
    binarize_volatility,
    convert_bin_list_to_str,
    convert_bin_str_to_list,
    get_binarization_params,
    unbinarize,
    unbinarize_df,
)


@pytest.fixture
def df_and_binarization_params(request):
    n_bits = request.param
    df = pd.DataFrame(
        {
            "a": np.arange(2 ** n_bits),
            "b": np.linspace(0, 10, 2 ** n_bits),
            "c": np.linspace(-10, 10, 2 ** n_bits),
        }
    )

    binarization_params = {}
    for column in df.columns:
        binarization_params[column] = {
            "n_bits": n_bits,
            "x_min": df[column].min(),
            "x_max": df[column].max(),
        }

    return df, binarization_params, n_bits


@pytest.mark.parametrize("df_and_binarization_params", range(1, 16), indirect=True)
def test_binarize(df_and_binarization_params):
    df, binarization_params, n_bits = df_and_binarization_params
    binary_range = np.array([bin(x)[2:].zfill(n_bits) for x in range(2 ** n_bits)])

    for column in df.columns:
        assert binarize(df[column].min(), **binarization_params[column]) == "0" * n_bits
        assert binarize(df[column].max(), **binarization_params[column]) == "1" * n_bits
        assert (binarize(df[column], **binarization_params[column]) == binary_range).all()


@pytest.mark.parametrize("df_and_binarization_params", range(1, 16), indirect=True)
def test_binarize_df(df_and_binarization_params):
    df, binarization_params, n_bits = df_and_binarization_params
    binary_range = np.array([bin(x)[2:].zfill(n_bits) for x in range(2 ** n_bits)])

    df_binarized = binarize_df(df, binarization_params)
    for column in df_binarized.columns:
        assert (df_binarized[column] == binary_range).all()


@pytest.mark.parametrize("df_and_binarization_params", [16], indirect=True)
def test_binarize_volatility(df_and_binarization_params):
    test_volatility, _, _ = df_and_binarization_params

    volatility_binarized = binarize_volatility(test_volatility)
    for column in test_volatility.columns:
        x = test_volatility[column]
        x_binarized = volatility_binarized[f"{column}_binary"]

        assert x_binarized.dtype == np.int8
        assert set(x_binarized) == set([0, 1])
        assert (x_binarized == (x > x.median())).all()

    assert all([column.endswith("_binary") for column in volatility_binarized.columns])


@pytest.mark.parametrize(
    "bin_str, bin_list",
    [
        ("0000", [0, 0, 0, 0]),
        ("1010", [1, 0, 1, 0]),
        ("1111", [1, 1, 1, 1]),
        ("10101010", [1, 0, 1, 0, 1, 0, 1, 0]),
    ],
)
def test_convert_bin_str_to_list(bin_str, bin_list):
    assert all(convert_bin_str_to_list(bin_str) == bin_list)


@pytest.mark.parametrize(
    "bin_str, bin_list",
    [
        ("0000", [0, 0, 0, 0]),
        ("1010", [1, 0, 1, 0]),
        ("1111", [1, 1, 1, 1]),
        ("10101010", [1, 0, 1, 0, 1, 0, 1, 0]),
    ],
)
def test_convert_bin_list_to_str(bin_str, bin_list):
    assert convert_bin_list_to_str(bin_list) == bin_str


@pytest.mark.parametrize("df_and_binarization_params", [16], indirect=True)
def test_get_binarization_params_without_ϵ(df_and_binarization_params):
    df, binarization_params_, n_bits = df_and_binarization_params

    binarization_params = get_binarization_params(df, n_bits)

    for column in df.columns:
        assert binarization_params[column]["n_bits"] == n_bits
        assert binarization_params[column]["x_min"] == df[column].min()
        assert binarization_params[column]["x_max"] == df[column].max()


@pytest.mark.parametrize("df_and_binarization_params", [16], indirect=True)
def test_get_binarization_params_with_ϵ(df_and_binarization_params):
    df, binarization_params_, n_bits = df_and_binarization_params
    ϵ = {}
    for i, column in enumerate(df.columns):
        ϵ[column] = {"min": -i / 10, "max": i / 10}

    binarization_params = get_binarization_params(df, n_bits, ϵ=ϵ)

    for column in df.columns:
        assert binarization_params[column]["n_bits"] == n_bits
        assert binarization_params[column]["x_min"] == df[column].min() - ϵ[column]["min"]
        assert binarization_params[column]["x_max"] == df[column].max() + ϵ[column]["max"]


@pytest.mark.parametrize("df_and_binarization_params", range(1, 16), indirect=True)
def test_unbinarize(df_and_binarization_params):
    df, binarization_params, n_bits = df_and_binarization_params
    binary_range = np.array([bin(x)[2:].zfill(n_bits) for x in range(2 ** n_bits)])

    for column in df.columns:
        assert np.isclose(
            unbinarize("0" * n_bits, **binarization_params[column]), df[column].min()
        )
        assert np.isclose(
            unbinarize("1" * n_bits, **binarization_params[column]), df[column].max()
        )
        assert np.isclose(
            unbinarize(binary_range, **binarization_params[column]), df[column]
        ).all()


@pytest.mark.parametrize("df_and_binarization_params", range(1, 16), indirect=True)
def test_unbinarize_df(df_and_binarization_params):
    df, binarization_params, n_bits = df_and_binarization_params
    binary_range = np.array([bin(x)[2:].zfill(n_bits) for x in range(2 ** n_bits)])

    df_binarized = pd.DataFrame({"a": binary_range, "b": binary_range, "c": binary_range})
    df["a_binary"] = [0 for i in range(df.shape[0])]
    df["b_binary"] = [1 for i in range(df.shape[0])]
    df_binarized["a_binary"] = ["0".zfill(n_bits) for i in range(df.shape[0])]
    df_binarized["b_binary"] = ["1".zfill(n_bits) for i in range(df.shape[0])]

    df_unbinarized = unbinarize_df(df_binarized, binarization_params)
    for column in df.columns:
        assert np.isclose(df_unbinarized[column], df[column]).all()
