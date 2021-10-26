from copy import deepcopy
from functools import reduce
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal


def _get_holidays(start_date, end_date, exchanges=["NYSE", "LSE"]):
    """
    Returns a set of holidays for the provided exchanges. List of exchanges available here:
    https://pandas-market-calendars.readthedocs.io/en/latest/usage.html.

    :param start_date: Datetime of when to start the window.
    :param end_date: Datetime of when to end the window.

    :returns: A set of holidays for the provided exchanges.
    """
    holidays = set()
    for exchange in exchanges:
        calendar = mcal.get_calendar(exchange)
        exchange_holidays = set(
            [
                holiday
                for holiday in calendar.holidays().holidays
                if holiday >= start_date.date() and holiday <= end_date.date()
            ]
        )
        holidays = holidays.union(exchange_holidays)

    return holidays


def load_raw_data(
    data_dir,
    datasets,
    start_date=datetime(1999, 1, 1),
    end_date=datetime(2019, 12, 31),
    min_volume=1,
    common_dates=True,
):
    """
    Loads the raw csv files from the data directory. The files must corresponds to
    the strings in the datasets list, i.e., if datasets = ["abc", "xyz"], then the files
    in the data directory must contain "abc" and "xyz" in their names.

    :param data_dir: Directory containing the raw csv files.
    :param datasets: List of strings corresponding to the file names (w/o extensions).
    :param start_date: Date at which to drop all data before (inclusive).
    :param end_date: Date at which to drop all data after (inclusive).
    :param min_volume: Filter out dates with volume less than the provided min_volume.
    :param common_dates: Ensure all dataframes have the same dates.

    :returns: Dictionary with the strings in datasets as keys and dataframes as values.
    """
    dfs = {}
    csvs = [x for x in data_dir.iterdir() if str(x).endswith(".csv")]
    for dataset in datasets:
        # load all csv files for each dataset
        dataset_csvs = [x for x in csvs if dataset in str(x)]
        dataset_dfs = []
        for csv in dataset_csvs:
            dataset_dfs.append(
                pd.read_csv(
                    data_dir / csv,
                    parse_dates=["Gmt time"],
                    date_parser=lambda x: pd.to_datetime(
                        x, format="%d.%m.%Y %H:%M:%S.%f"
                    ),
                    index_col="Gmt time",
                )
            )
        df = pd.concat(dataset_dfs, axis=0)

        # convert column names to lower case w/o spaces
        columns = df.columns.to_list()
        column_map = {column: column.lower().replace(" ", "_") for column in columns}
        df.rename(columns=column_map, inplace=True)
        df.index.rename("date", inplace=True)

        # calculate return columns
        df["return"] = (df["close"] - df["open"]) / df["open"]
        df["log_return"] = np.log(df["close"] / df["open"])

        # drop days w/ volume <= 0, weekends, and holidays
        holidays = _get_holidays(start_date, end_date)
        df = df.loc[
            (df["volume"] >= min_volume)
            & (df.index >= start_date)
            & (df.index <= end_date)
            & (df.index.day_of_week < 5)
            & (~df.index.isin(holidays))
        ]
        assert (df["volume"] >= min_volume).all()
        assert (df.index >= start_date).all()
        assert (df.index <= end_date).all()
        assert (~df.index.isin(holidays)).all()

        # add to dict of dfs
        dfs[dataset] = df

    # filter the dataframes to have a common set of dates
    if common_dates:
        for i, dataset in enumerate(datasets):
            if i == 0:
                common_dates = set(dfs[dataset].index)
            else:
                common_dates = common_dates.intersection(set(dfs[dataset].index))

        for dataset in datasets:
            dfs[dataset] = dfs[dataset].loc[common_dates].sort_index()

        shapes = np.array([df.shape[0] for df in dfs.values()])
        assert (shapes == shapes[0]).all()

    return dfs


def merge_dfs(dfs, datasets):
    """
    Merges the dataframes into one, prefixing the columns with the keys of dfs.

    :param dfs: Dictionary with strings as keys and dataframes as values.

    :returns: Merged dataframe.
    """
    dfs = deepcopy(dfs)
    for dataset in datasets:
        # prefix column names with dataset
        columns = dfs[dataset].columns.to_list()
        column_map = {column: f"{dataset}_{column}" for column in columns}
        dfs[dataset].rename(columns=column_map, inplace=True)

    # merge the dfs
    df = reduce(
        lambda X, Y: pd.merge(X, Y, how="inner", left_index=True, right_index=True),
        list(dfs.values()),
    ).dropna()

    assert len(df.columns) == len(dfs) * len(dfs[datasets[0]].columns)

    return df
