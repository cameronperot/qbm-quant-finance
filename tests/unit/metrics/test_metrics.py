import numpy as np
import pandas as pd
import pytest

from qbm.metrics import (
    compute_annualized_volatility,
    compute_correlation_coefficients,
    compute_rolling_volatility,
)


@pytest.fixture
def df():
    return pd.DataFrame({"a": np.linspace(0, 10, 100), "b": np.linspace(-10, 10, 100)})


def test_compute_annualized_volatility(df):
    annualized_volatility = compute_annualized_volatility(df)

    assert annualized_volatility.equals(df.std() * np.sqrt(252))


def test_compute_correlation_coefficients(monkeypatch, df):
    combinations = [("a", "b"), ("b", "a")]
    combination_indices = [f"{x}/{y}" for x, y in combinations]
    return_values = {"Pearson": 0, "Spearman": 1, "Kendall": 2}

    monkeypatch.setattr(
        "qbm.metrics.metrics.pearsonr", lambda x, y: (return_values["Pearson"], None)
    )
    monkeypatch.setattr(
        "qbm.metrics.metrics.spearmanr", lambda x, y: (return_values["Spearman"], None)
    )
    monkeypatch.setattr(
        "qbm.metrics.metrics.kendalltau", lambda x, y: (return_values["Kendall"], None)
    )

    correlation_coefficients = compute_correlation_coefficients(df, combinations)

    assert set(correlation_coefficients.index) == set(combination_indices)
    assert set(correlation_coefficients.columns) == set(return_values.keys())
    for column, return_value in return_values.items():
        assert (correlation_coefficients[column] == return_value).all()


def test_compute_correlation_coefficients_sample_index(monkeypatch, df):
    combinations = [("a", "b"), ("b", "a")]
    return_values = {"Pearson": 0, "Spearman": 1, "Kendall": 2}

    monkeypatch.setattr(
        "qbm.metrics.metrics.pearsonr", lambda x, y: (return_values["Pearson"], None)
    )
    monkeypatch.setattr(
        "qbm.metrics.metrics.spearmanr", lambda x, y: (return_values["Spearman"], None)
    )
    monkeypatch.setattr(
        "qbm.metrics.metrics.kendalltau", lambda x, y: (return_values["Kendall"], None)
    )

    correlation_coefficients = compute_correlation_coefficients(df, combinations, 1)

    assert (correlation_coefficients["sample"] == 1).all()


def test_compute_rolling_volatility_keep_rows(monkeypatch, df):
    window = 10
    renamed_columns = [f"{column}_volatility" for column in df.columns]

    volatility = compute_rolling_volatility(df, window, drop_insufficient_data_rows=False)

    assert (volatility.index == df.index).all()
    assert set(volatility.columns) == set(renamed_columns)


def test_compute_rolling_volatility_drop_rows(monkeypatch, df):
    window = 10
    renamed_columns = [f"{column}_volatility" for column in df.columns]

    volatility = compute_rolling_volatility(df, window, drop_insufficient_data_rows=True)

    assert (volatility.index >= window).all()
    assert set(volatility.columns) == set(renamed_columns)
