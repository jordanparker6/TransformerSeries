from typing import List, Optional

class PositionEncoding:
    pass

class TimeseriesWindowBuilder:
    def __init__(self, training_length: int, forecast_window: int):
        self.training_length = training_length
        self.forecast_window = forecast_window

    def fit(self, dataset)
        X = []
        Y = []

        df = self.df.loc[: , self.features]
        unique_groups = df.group_id.unique()

        for group in unique_groups:
            tmp = dataset._df[df.group_id == group].values
            tmp_target = df.loc[df.group_id == group, dataset.targets].values
            
            itr = range(0, tmp.shape[0] - self.training_length - self.forecast_window, self.training_length + self.forecast_window)
            for i in itr:
                src = tmp[i: i + self.training_length].tolist()
                trg = [[0]] + tmp_target[i + self.training_length : i + self.training_length + self.forecast_window].tolist()

                X.append(src)
                Y.append(trg)
                
        self.X = np.array(X, dtype=np.float64)
        self.Y = np.array(Y, dtype=np.float64)