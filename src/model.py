import math
import torch
import torch.nn as nn
import config

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
        num_layers: int = config.MODEL["num_layers"],
        attn_heads: int = config.MODEL["attn_heads"],
        dropout: float = config.MODEL["dropout"]
    ):
        """
        Args:
            feature_size (int, optional): [description]. Defaults to 7.
            num_layers (int, optional): The number of encoding layers. Defaults to 3.
            attn_heads (int, optional): The number of attention heads at each layer. Defaults to 8.
            dropout (float, optional): The dropout probability. Defaults to 0.
        """
        super(TransformerSeries, self).__init__()
        ## ADD LAYER NORMALISATION
        self.model_type = 'Transformer'
        self.layer_norm = nn.LayerNorm(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=attn_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers, norm=self.layer_norm)        
        self.decoder_1 = nn.Linear(feature_size, 2048)
        self.decoder_2 = nn.Linear(2048, output_size) 
        self.init_weights()

    def init_weights(self, initrange: float = 0.1):
        """Initiates weight variables. ~Uniform(-initrage, initrange)

        Args:
            initrange (float, optional): The initial weight range +/-. Defaults to 0.1.
        """
        self.decoder_1.bias.data.zero_()
        self.decoder_1.weight.data.uniform_(-initrange, initrange)
        self.decoder_2.bias.data.zero_()
        self.decoder_2.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, size: int):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, X, device):  
        mask = self._generate_square_subsequent_mask(len(X)).to(device)
        output = self.transformer_encoder(X, mask)
        output = self.decoder_1(output)
        output = self.decoder_2(output)
        return output