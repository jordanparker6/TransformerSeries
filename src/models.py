import math
import torch
import torch.nn as nn
import config

class Baseline(nn.Module):
    """
    A naive baseline model for benchmarking.

    The baseline model will learn a linear combination of the input features.
    """
    def __init__(self, feature_size: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(feature_size, output_size)

    def forward(self, X):
        assert len(X.shape) == 3, "Tensor must be of form [nsamples, batch, features]."
        return self.linear(X)

class TransformerSeries(nn.Module):
    """
    A Transformer for timeseries data.

    The standard Trasnsformer encoder layer is based on the paper “Attention Is All You Need”.
    It implements multi-headed self-attention. The TransformerEncoder stacks the encoder layer and
    implements layer normalisation (optional). The decoder is replaced by a FNN, a step that has become
    fashionable since the original paper.
    """
    def __init__(self, 
        feature_size: int,
        output_size: int,
        num_layers: int = config.MODEL["transformer"]["num_layers"],
        attn_heads: int = config.MODEL["transformer"]["attn_heads"],
        dropout: float = config.MODEL["dropout"],
    ):
        """
        Args:
            feature_size (int, optional): [description]. Defaults to 7.
            num_layers (int, optional): The number of encoding layers. Defaults to 3.
            attn_heads (int, optional): The number of attention heads at each layer. Defaults to 8.
            dropout (float, optional): The dropout probability. Defaults to 0.
        """
        super(TransformerSeries, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=attn_heads, dropout=dropout), 
            num_layers=num_layers, 
            norm=nn.LayerNorm(feature_size)
        )        
        self.decoder = nn.Linear(feature_size, 2048)
        self.final_layer = nn.Linear(2048, output_size) 
        self.init_weights()

    def init_weights(self, initrange: float = 0.1):
        """Initiates weight variables. ~Uniform(-initrage, initrange)

        Args:
            initrange (float, optional): The initial weight range +/-. Defaults to 0.1.
        """
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.final_layer.bias.data.zero_()
        self.final_layer.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): shape --> [input_size, batch, feature_size]
        """
        assert len(X.shape) == 3, "Tensor must be of form [nsamples, batch, features]."
        mask = self._generate_square_subsequent_mask(len(X)).to(config.DEVICE)
        output = self.transformer_encoder(X, mask)
        output = self.decoder(output)
        return self.final_layer(output)

class LSTM(nn.Module):
    """
    A LSTM for timeseries data.

    This implements a vanila LSTM model with a dense final layer.
    """
    def __init__(self,
            feature_size: int, 
            output_size: int,
            batch_size: int = config.DATASET["batch_size"],
            hidden_states: int = config.MODEL["lstm"]["hidden_states"],
            layers: int = config.MODEL["lstm"]["layers"],
            bias: bool = True,
        ):
        super(LSTM, self).__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.hidden_states = hidden_states
        self.lstm_layers = layers
        self.batch_size = batch_size
        self.lstm = torch.nn.LSTM(feature_size, hidden_states, layers, bias)
        self.final_layer = nn.Linear(hidden_states, output_size)
        self.hidden = self.init_hidden(batch_size)

    def init_hidden(self, batch_size: int):
        return (
                torch.zeros(self.lstm_layers, batch_size, self.hidden_states),  # h_n hidden state
                torch.zeros(self.lstm_layers, batch_size, self.hidden_states)   # c_n final cell state
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

ALL = {
    "transformer": TransformerSeries,
    "lstm": LSTM,
    "baseline": Baseline,
}