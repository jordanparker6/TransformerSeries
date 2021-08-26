from .base import BaseModel

class Baseline(LitBase):
    """
    A naive baseline model for benchmarking.

    The baseline model will learn a linear combination of the input features.
    """
    def __init__(self, dataset: TimeSeriesDataset):
        super().__init__(dataset)
        self.linear = nn.Linear(self.feature_size, self.output_size)

    def forward(self, X):
        assert len(X.shape) == 3, "Tensor must be of form [nsamples, batch, features]."
        return self.linear(X)