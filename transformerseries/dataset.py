import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import joblib
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing

import config

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
        position_encoded = ["sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month", "cos_month"]

        assert "timestamp" in df.columns, "Column 'timestamp' does not exist. Please ensure the dataset has been preprocessed."
        assert "group_id" in df.columns, "Column 'group_id' does not exist. Please ensure the dataset has been preprocessed."
        assert all(x in df.columns for x in targets), "A target column doesn't exist in the dataset."
        assert all(x in df.columns for x in position_encoded), "The target dataset has not been position encoded. Please ensure the dataset has been preprocessed"
       
        df = df.drop(columns=["timestamp"])

        le = preprocessing.LabelEncoder()
        df["group_id"] = le.fit_transform(df.group_id)
        self.le = le

        scaler = MinMaxScaler()  
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

        self.df = df
        self.scaler = scaler                  # could make this also NormScaler
        self.training_length = training_length
        self.forecast_window = forecast_window
        
        cols = [col for col in self.df.columns if col not in ["group_id"]]
        self.raw_features = [col for col in cols if col not in position_encoded]
        self.features = self.raw_features + position_encoded
        self.targets = targets

        self.build_dataset()
    
    def build_dataset(self):
        X = []
        Y = []

        df = self.df.loc[: , self.features]
        unique_groups = df.group_id.unique()

        for group in unique_groups:

            tmp = df[df.group_id == group].values
            itr = range(0, tmp.shape[0], self.training_length + self.forecast_window)
            for i in itr:

                src = tmp[i: i+self.training_length]
                trg = tmp[i+self.training_length: i+self.training_length+self.forecast_window]

                X.append(src)
                Y.append(trg)

        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    """
    def transform(self, X, Y):
        Transforms the scale of the X, y with X to avoid target leakage.
        p = len(self.raw_features)
        self.scaler.fit(X[:, 0:p])
        X[:, 0:p] = torch.tensor(self.scaler.transform(X[:, 0:p]))
        Y[:, 0:p] = torch.tensor(self.scaler.transform(Y[:, 0:p]))
        # save the scalar to be used later when inverse translating the data for plotting.
        joblib.dump(self.scaler, 'scalar_item.joblib')
        return X, Y

    def inverse_transform(self, X, scaler):
        Rescales a tensor by the transform scaler. Adds padding to tensors not of the original input size
        n, shape = len(self.raw_features), X.shape
        assert len(shape) == 3, "Tensor must be of form [nsamples, batch, features]."

        X = X.detach().reshape(shape[0], shape[2])[:, 0:n].cpu()
        if shape[2] < n:
            padding = torch.zeros((shape[0], n - shape[2]))
            X = torch.cat((X, padding), dim=1)
        
        assert X.shape[1] == n, "Dim 2 does not equal feature size."
        return scaler.inverse_transform(X.cpu().data.numpy())
    """


    def get_parameters(self):
        return {
            "targets": self.targets,
            "features": self.features,
            "raw_features": self.raw_features,
            "training_length": self.training_length,
            "forecast_window": self.forecast_window,
            "groups": self.groups,
            "dates": self.dates
        }