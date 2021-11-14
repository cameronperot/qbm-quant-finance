import numpy as np
import pandas as pd
import pytest

from datetime import datetime, timedelta
from pathlib import Path

from qbm.utils import load_log_returns


@pytest.fixture
def log_returns_df():
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2020, 12, 31)
    dates = []
    date = start_date
    while date <= end_date:
        dates.append(date)
        date += timedelta(days=1)

    log_returns = pd.DataFrame(
        {
            "EURUSD": np.linspace(-1, 1, len(dates)),
            "GBPUSD": np.linspace(-2, 2, len(dates)),
            "USDCAD": np.linspace(-3, 3, len(dates)),
            "USDJPY": np.linspace(-4, 4, len(dates)),
        },
        index=dates,
    )

    return log_returns


def test_load_log_returns(monkeypatch, log_returns_df):
    monkeypatch.setattr("qbm.utils.data_loading.get_project_dir", lambda: Path("test_dir"))
    monkeypatch.setattr(
        "qbm.utils.data_loading.pd.read_csv",
        lambda file_path, parse_dates, index_col: log_returns_df,
    )

    start_date = datetime(2005, 1, 1)
    end_date = datetime(2015, 12, 31)
    outlier_threshold = 1

    log_returns_with_outliers = load_log_returns(
        "test", start_date=start_date, end_date=end_date, outlier_threshold=None
    )
    log_returns_without_outliers = load_log_returns(
        "test",
        start_date=start_date,
        end_date=end_date,
        outlier_threshold=outlier_threshold,
    )

    for pair in log_returns_df.columns:
        x = log_returns_with_outliers[pair].copy()
        x = x[((x - x.mean()) / x.std()).abs() <= outlier_threshold]
        assert np.isclose(log_returns_without_outliers[pair], x).all()

    assert (log_returns_with_outliers.index >= start_date).all()
    assert (log_returns_with_outliers.index <= end_date).all()
    assert (log_returns_without_outliers.index >= start_date).all()
    assert (log_returns_without_outliers.index <= end_date).all()
