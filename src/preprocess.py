import argparse
import ast
import pandas as pd
import numpy as np
from typing import List
from pathlib import Path

import config

class TimeSeriesPreprocessor:
    """
    Timeseries data pre-processor.
    
    Returns a train / test dataset in ascending time order and date position encoding.
    """
    def __init__(self, data_dir: Path):
        self.dir = data_dir
        self.test_split = 0.2

    def __call__(self, 
            csv_file: str, 
            date_column: str, 
            group_columns: List = [],
            frequency: str = "daily"
        ):
        df = pd.read_csv(csv_file, parse_dates=[date_column])
        df = self.add_time_id_by_frequency(df, date_column, frequency)
        df = self.sinusoidal_position_encoding(df, date_column)
        df = df.sort_index(ascending=True)
        
        #train, test = self.test_train_split_by_groups(df, group_columns)
        #train.to_csv(self.dir.joinpath("train_raw.csv"))
        #test.to_csv(self.dir.joinpath("test_raw.csv"))
        group_columns = [x for x in group_columns if x != "group_id"]
        #train = train.drop(columns=group_columns)
        #test = test.drop(columns=group_columns)
        #train.to_csv(self.dir.joinpath("train.csv"))
        #test.to_csv(self.dir.joinpath("test.csv"))
        df.to_csv(self.dir.joinpath("data.csv"))
        return df

    def sinusoidal_position_encoding(self, df: pd.DataFrame, date_column: str):
        """
        Encodes the position of hour, day, month seaonality. RBF could be used in place.
        https://medium.com/mlearning-ai/transformer-implementation-for-time-series-forecasting-a9db2db5c820
        """

        # Encode Date Positioning
        hour = df[date_column].dt.hour / 24
        day = df[date_column].dt.day / 30.5
        month = df[date_column].dt.month / 12
        year = df[date_column].dt.year
        df['sin_hour'] = np.sin(2 * np.pi * hour)
        df['cos_hour'] = np.cos(2 * np.pi * hour)
        df['sin_day'] = np.sin(2 * np.pi * day)
        df['cos_day'] = np.cos(2 * np.pi * day)
        df['sin_month'] = np.sin(2 * np.pi * month)
        df['cos_month'] = np.cos(2 * np.pi * month)
        df['year'] = year
        # could also add day of week, week of year

        # Set Date Index
        df['timestamp'] = df[date_column]
        if date_column != "timestamp":
            df = df.drop(columns=[date_column])
        df = df.set_index("timestamp")
        return df

    def add_time_id_by_frequency(self, df: pd.DataFrame, date_column: str, frequency: str):
        date = df[date_column].dt
        if frequency == "monthly":
            df["time_idx"] = date.year * 12 + date.month
        elif frequency == "weekly":
            df["time_idx"] = date.year * 52 + date.weekofyear
        elif frequency == "daily":
            df["time_idx"] = date.year * 365 + date.dayofyear
        else:
            raise NotImplementedError
        df["time_idx"] -= df["time_idx"].min()
        return df
        
    def test_train_split_by_groups(self, df: pd.DataFrame, group_columns: List):
        """Creates a timeseries test / train split for each group."""
        train, test = pd.DataFrame(), pd.DataFrame()
        groups = [("_", df)]
        if group_columns:
            groups = df.groupby(group_columns)

        for i, groupby in enumerate(groups):
            name, group = groupby
            n = len(group)
            df["group_id"] = i
            group = group.sort_index(ascending=True)
            train_i = range(0, int(n * (1 - self.test_split)))
            test_i = range(int(n * (1 - self.test_split)) + 1, n)
            train = pd.concat((train, group.iloc[train_i, :]))
            test = pd.concat((test, group.iloc[test_i, :]))

        return train, test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLI to execute data preprocessing.')
    parser.add_argument("--csv_file", type=str)
    parser.add_argument("--date_column", type=str, default="date")
    parser.add_argument("--group_columns", type=str, default="[]")
    parser.add_argument("--frequency", type=str, default="daily")

    args = parser.parse_args()

    processor = TimeSeriesPreprocessor(config.DATA_DIR)
    data = processor(
            args.csv_file, 
            args.date_column,
            ast.literal_eval(args.group_columns),
            args.frequency
        )
    print(data.head())