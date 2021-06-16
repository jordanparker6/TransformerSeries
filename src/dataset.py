import argparse
import pandas as pd
import numpy as np
import ast
from pathlib import Path
from typing import List
import joblib
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

import config

#########################
## Preprocessing ########
#########################

class TimeSeriesPreprocessor:
    """
    Timeseries data pre-processor.
    
    Returns a train / test dataset in ascending time order and date position encoding.
    """
    def __init__(self, data_dir: Path):
        self.dir = data_dir
        self.test_split = 0.2

    def __call__(self, csv_file: str, date_column: str, group_columns: List = []):
        df = pd.read_csv(csv_file, parse_dates=[date_column])
        df = self.sinusoidal_position_encoding(df, date_column)
        train, test = self.test_train_split_by_groups(df, group_columns)
        train.to_csv(self.dir.joinpath("train_raw.csv"))
        test.to_csv(self.dir.joinpath("test_raw.csv"))
        train = train.drop(columns=group_columns)
        test = test.drop(columns=group_columns)
        train.to_csv(self.dir.joinpath("train.csv"))
        test.to_csv(self.dir.joinpath("test.csv"))
        return train, test

    def sinusoidal_position_encoding(self, df: pd.DataFrame, date_column: str):
        """
        Encodes the position of hour, day, month seaonality. RBF could be used in place.
        https://medium.com/mlearning-ai/transformer-implementation-for-time-series-forecasting-a9db2db5c820
        """
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
        df['timestamp'] = df[date_column]
        df = df.drop(columns=[date_column]).set_index("timestamp")
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
 
#########################
## Dataset ##############
#########################

class TimeSeriesDataset(Dataset):
    def __init__(
        self, 
        csv_file: Path,
        targets: List[str],
        training_length: int, 
        forecast_window: int,
    ):  
        """
        Args:
            csv_file (Path): The path object for the csv file.
            targets: The column names of the variable being predicted.
            training_length (int): The length of the sequence included in the training.
            forecast_window (int): The forecast window for predictions.
        """
        df = pd.read_csv(csv_file, parse_dates=["timestamp"]).reset_index(drop=True)
        self.dates = df["timestamp"]
        self.df = df.drop(columns=["timestamp"])
        self.scaler = MinMaxScaler()                    # could make this also NormScaler
        self.T = training_length
        self.S = forecast_window
        
        cols = [col for col in self.df.columns if col not in ["group_id"]]
        position_encoded = ["sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month", "cos_month"]
        self.raw_features = [col for col in cols if col not in position_encoded]
        self.features = self.raw_features + position_encoded
        self.targets = targets

    def __len__(self):
        """Returns the number or groups."""
        return len(self.df.groupby(by=["group_id"]))

    def __getitem__(self, idx):
        """
        Randomly pulls a timeseries for each group.
        The lenght of the timeseries X, Y is the training length and forecast window.
        """
        start = np.random.randint(0, len(self.df[self.df["group_id"] == idx]) - self.T - self.S) 

        group_df = self.df.loc[self.df["group_id"] == idx, self.features]
        group = self.df.loc[self.df["group_id"] == idx, ["group_id"]][ start : start + 1].values.item()
        
        X = group_df[start : start + self.T]
        Y = group_df[start + self.T : start + self.T + self.S]

        X_i = X.index.values
        Y_i = Y.index.values
        X = torch.tensor(X.values)
        Y = torch.tensor(Y.values)

        X, Y = self.transform(X, Y)
        
        return X_i, Y_i, X, Y, group

    def transform(self, X, Y):
        """Transforms the scale of the X, y with X to avoid target leakage."""
        p = len(self.raw_features)
        self.scaler.fit(X[:, 0:p])
        X[:, 0:p] = torch.tensor(self.scaler.transform(X[:, 0:p]))
        Y[:, 0:p] = torch.tensor(self.scaler.transform(Y[:, 0:p]))
        # save the scalar to be used later when inverse translating the data for plotting.
        joblib.dump(self.scaler, 'scalar_item.joblib')
        return X, Y

    def inverse_transform(self, X, scaler):
        n, shape = len(self.raw_features), X.shape
        assert len(shape) == 3
        X = X[:, :, 0:n].detach().squeeze()
        if shape[2] < n:
            padding = torch.zeros((shape[0], n - shape[2]))
            X = torch.cat((X, padding), dim=1).to(config.DEVICE)
        return scaler.inverse_transform(X) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLI to execute data preprocessing.')
    parser.add_argument("--csv_file", type=str)
    parser.add_argument("--date_column", type=str, default="date")
    parser.add_argument("--group_columns", type=str, default="[]")
    args = parser.parse_args()

    processor = TimeSeriesPreprocessor(config.DATA_DIR)
    train, test = processor(
            args.csv_file, 
            args.date_column,
            ast.literal_eval(args.group_columns)
        )
    print(train.head())
    print(test.head())
