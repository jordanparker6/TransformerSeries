from .base import BaseModel

class TransformerSeries(LitBase):
    """
    A Transformer for timeseries data.

    The standard Trasnsformer encoder layer is based on the paper “Attention Is All You Need”.
    It implements multi-headed self-attention. The TransformerEncoder stacks the encoder layer and
    implements layer normalisation (optional). The decoder is replaced by a FNN, a step that has become
    fashionable since the original paper.
    """
    def __init__(self, 
        dataset: TimeSeriesDataset,
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
        super(TransformerSeries, self).__init__(dataset)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.feature_size, nhead=attn_heads, dropout=dropout), 
            num_layers=num_layers, 
            norm=nn.LayerNorm(self.feature_size)
        )        
        self.decoder = nn.Linear(self.feature_size, 2048)
        self.final_layer = nn.Linear(2048, self.output_size) 
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
        mask = (torch.triu(torch.ones(size, size, device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): shape --> [input_size, batch, feature_size]
        """
        assert len(X.shape) == 3, "Tensor must be of form [nsamples, batch, features]."
        mask = self._generate_square_subsequent_mask(len(X))
        output = self.transformer_encoder(X, mask)
        output = self.decoder(output)
        return self.final_layer(output)