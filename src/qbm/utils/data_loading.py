from copy import deepcopy
from functools import reduce
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal


def load_log_returns(file_path):
    """
    Loads the training data (log returns) in float form.

    :param file_path: Path to the log_returns.csv file.

    :returns: Log returns dataframe.
    """
    return pd.read_csv(file_path, parse_dates=["date"], index_col="date",)


def load_raw_data(
    data_dir,
    data_source,
    start_date=datetime(1999, 1, 1),
    end_date=datetime(2019, 12, 31),
    min_volume=1,
    common_dates=True,
    drop_holidays=True,
):
    """
    Loads the raw csv files from the data directory. The files must corresponds to
    the strings in the currency_pairs list, i.e., if currency_pairs = ["abc", "xyz"], then the files
    in the data directory must contain "abc" and "xyz" in their names.

    :param data_dir: Directory containing the raw csv files.
    :param currency_pairs: List of strings corresponding to the file names (w/o extensions).
    :param start_date: Date at which to drop all data before (inclusive).
    :param end_date: Date at which to drop all data after (inclusive).
    :param min_volume: Filter out dates with volume less than the provided min_volume.
    :param common_dates: Ensure all dataframes have the same dates.
    :param drop_holidays: Drop dates that are NYSE or LSE holidays.

    :returns: Dictionary with the strings in currency_pairs as keys and dataframes as values.
    """
    currency_pairs = ["EURUSD", "GBPUSD", "USDCAD", "USDJPY"]

    if data_source == "dukascopy":
        data_dir /= "dukascopy"
        csvs = [x for x in data_dir.iterdir() if str(x).endswith(".csv")]

        dfs = {}
        for pair in currency_pairs:
            # load all csv files for each pair
            pair_csvs = [x for x in csvs if pair in str(x)]
            pair_dfs = []
            for csv in pair_csvs:
                pair_dfs.append(
                    pd.read_csv(
                        data_dir / csv,
                        parse_dates=["Gmt time"],
                        date_parser=lambda x: pd.to_datetime(
                            x, format="%d.%m.%Y %H:%M:%S.%f"
                        ),
                        index_col="Gmt time",
                    )
                )
            df = pd.concat(pair_dfs, axis=0)

            # convert column names to lower case w/o spaces
            columns = df.columns.to_list()
            column_map = {column: column.lower().replace(" ", "_") for column in columns}
            df.rename(columns=column_map, inplace=True)
            df.index.rename("date", inplace=True)

            # calculate return columns
            df["return"] = (df["close"] - df["open"]) / df["open"]
            df["log_return"] = np.log(df["close"] / df["open"])

            # drop days w/ volume <= 0, weekends, and holidays
            if drop_holidays:
                holidays = _get_holidays(start_date, end_date)
            else:
                holidays = set()

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

            # add pair df to dict of dfs
            dfs[pair] = df

        # filter the dataframes to have a common set of dates
        if common_dates:
            for i, pair in enumerate(currency_pairs):
                if i == 0:
                    common_dates = set(dfs[pair].index)
                else:
                    common_dates = common_dates.intersection(set(dfs[pair].index))

            for pair in currency_pairs:
                dfs[pair] = dfs[pair].loc[common_dates].sort_index()

            shapes = np.array([df.shape[0] for df in dfs.values()])
            assert (shapes == shapes[0]).all()

        log_returns = pd.DataFrame(
            {pair: dfs[pair]["log_return"] for pair in currency_pairs}
        )

        return dfs, log_returns

    elif data_source == "kaggle":
        data_dir /= "kaggle"

        # load raw data
        df = pd.read_csv(
            data_dir / "Foreign_Exchange_Rates.csv",
            parse_dates=["Time Serie"],
            date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d"),
            index_col="Time Serie",
            na_values="ND",
        ).dropna()
        df.index.rename("date", inplace=True)

        # create pair columns
        df["EURUSD"] = 1 / df["EURO AREA - EURO/US$"]
        df["GBPUSD"] = 1 / df["UNITED KINGDOM - UNITED KINGDOM POUND/US$"]
        df["USDCAD"] = df["CANADA - CANADIAN DOLLAR/US$"]
        df["USDJPY"] = df["JAPAN - YEN/US$"]

        # filter columns
        df = df.loc[:, currency_pairs]

        # filter rows
        if drop_holidays:
            holidays = _get_holidays(start_date, end_date)
        else:
            holidays = set()

        df = df.loc[
            (df.index >= start_date)
            & (df.index <= end_date)
            & (df.index.day_of_week < 5)
            & (~df.index.isin(holidays))
        ]

        assert (df.index >= start_date).all()
        assert (df.index <= end_date).all()
        assert (~df.index.isin(holidays)).all()

        # compute the log returns
        log_returns = df.iloc[:-1].copy()
        for pair in currency_pairs:
            log_returns[pair] = np.log(df[pair][1:].to_numpy() / df[pair][:-1].to_numpy())

        return df, log_returns


def merge_dfs(dfs, currency_pairs):
    """
    Merges the dataframes into one, prefixing the columns with the keys of dfs.

    :param dfs: Dictionary with strings as keys and dataframes as values.
    :param currency_pairs: List of currency pairs in dataset, e.g.
        ["EURUSD", "GBPUSD", "USDCAD", "USDJPY"]

    :returns: Merged dataframe.
    """
    dfs = deepcopy(dfs)
    for pair in currency_pairs:
        # prefix column names with pair
        columns = dfs[pair].columns.to_list()
        column_map = {column: f"{pair}_{column}" for column in columns}
        dfs[pair].rename(columns=column_map, inplace=True)

    # merge the dfs
    df = reduce(
        lambda X, Y: pd.merge(X, Y, how="inner", left_index=True, right_index=True),
        list(dfs.values()),
    ).dropna()

    assert len(df.columns) == len(dfs) * len(dfs[currency_pairs[0]].columns)

    return df


def _get_holidays(start_date, end_date, exchanges=["NYSE", "LSE"]):
    """
    Returns a set of holidays for the provided exchanges. List of exchanges available here:
    https://pandas-market-calendars.readthedocs.io/en/latest/usage.html.

    :param start_date: Datetime of when to start the window.
    :param end_date: Datetime of when to end the window.
    :param exchanges: List of exchanges to use holidays of.

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
