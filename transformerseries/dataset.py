import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import joblib
import torch
from torch.utils.data import Dataset

from core import config
from processors.base import ProcessPipeline

class TimeSeriesDataset(Dataset):
    def __init__(
        self, 
        csv_file: Path,
        targets: List[str],
        training_length: int, 
        forecast_window: int,
        processor: ProcessPipeline = None,
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

        if le == None:
            le = preprocessing.LabelEncoder()
            df["group_id"] = le.fit_transform(df.group_id)
        else:
            df["group_id"] = le.transform(df.group_id)
   
        if scaler == None:
            scaler = MinMaxScaler()  
            df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        else:
            df = pd.DataFrame(scaler.transform(df), columns=df.columns)

        self.le = le
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
        """Apply feature engineering steps and creates test / train split."""
        X = []
        Y = []

        df = self.df.loc[: , self.features]
        unique_groups = df.group_id.unique()
        self.groups = unique_groups

        for group in unique_groups:

            tmp = df[df.group_id == group].values
            tmp_target = df.loc[df.group_id == group, self.targets].values
            
            itr = range(0, tmp.shape[0] - self.training_length - self.forecast_window, self.training_length + self.forecast_window)
            for i in itr:

                src = tmp[i: i+self.training_length].tolist()
                trg = [[0]] + tmp_target[i+self.training_length: i+self.training_length+self.forecast_window].tolist()

                X.append(src)
                Y.append(trg)
                
        self.X = np.array(X, dtype=np.float64)
        self.Y = np.array(Y, dtype=np.float64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

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
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.scaler.transform(df), columns=df.columns)
    
    def inverse_transform(self, out: np.ndarray) -> List[float]:
        
        #Create an empty dataframe, same shape as training dataframe, only fill 
        #the target index (in this case 1), scale it, then only retrieve the target
        vals = np.empty((out.shape[0] ,9))
        vals[:, 1] = out.reshape(-1)

        return self.scaler.inverse_transform(vals)[:, 1].tolist()