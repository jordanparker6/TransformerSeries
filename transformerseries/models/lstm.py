from .base import BaseModel

class LSTM(BaseModel):
    """
    A LSTM for timeseries data.

    This implements a vanila LSTM model with a dense final layer.
    """
    def __init__(self,
            dataset: TimeSeriesDataset,
            batch_size: int = config.DATASET["batch_size"],
            hidden_states: int = config.MODEL["lstm"]["hidden_states"],
            layers: int = config.MODEL["lstm"]["layers"],
            bias: bool = True,
        ):
        super(LSTM, self).__init__(dataset)
        self.hidden_states = hidden_states
        self.lstm_layers = layers
        self.batch_size = batch_size
        self.lstm = torch.nn.LSTM(self.feature_size, hidden_states, layers, bias)
        self.final_layer = nn.Linear(hidden_states, self.output_size)
        self.hidden = self.init_hidden(batch_size)

    def init_hidden(self, batch_size: int):
        return (
                torch.zeros(self.lstm_layers, batch_size, self.hidden_states, device=self.device),  # h_n hidden state
                torch.zeros(self.lstm_layers, batch_size, self.hidden_states, device=self.device)  # c_n final cell state
            )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): shape --> [input_size, batch, feature_size]
        """
        assert len(X.shape) == 3, "Tensor must be of form [nsamples, batch, features]."
        hn, cn = self.hidden
        output, self.hidden = self.lstm(X, (hn.detach().double(), cn.detach().double()))
        return self.final_layer(output)