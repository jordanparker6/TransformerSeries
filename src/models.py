import math
import torch
from torch import nn
import config
from pytorch_lightning import LightningModule

from dataset import TimeSeriesDataset

#########################
## Utility Functions ####
#########################

def coin_flip(p):
    return True if torch.rand(1) < p else False

def sigmoid_decay(x: float, scale: float):
    return scale / (scale + torch.exp(x / scale))

####################################
## Base Pytorch Lightning Class ####
####################################

class LitBase(LightningModule):
    def __init__(self,
        dataset: TimeSeriesDataset,
        initial_teacher_period: int = config.MODEL["initial_teacher_period"],
        teacher_sampling_decay: int = config.MODEL["teacher_sampling_decay"]
    ):
        super().__init__()
        self.training_params = {
                "initial_teacher_period": initial_teacher_period,
                "teacher_sampling_decay": teacher_sampling_decay
            }
        self.dataset_params = dataset.get_parameters()
        self.loss = config.METRICS[config.MODEL["loss"]]
        self.feature_size = len(self.dataset_params["features"])
        self.output_size = len(self.dataset_params["targets"])

    def training_step(self, batch, batch_idx):
        X_i, Y_i, _X, _Y, group = batch
        X = _X.permute(1,0,2)[:-1,:,:]  # training data shifted left by 1.
        Y = _X.permute(1,0,2)[1:,:,:]   # training shifted right by 1.
        sampled_X = X[:1, :, :]         # init training sample as the first true sample  

        for i in range(len(Y)-1):
            pred = self.forward(sampled_X)

            # if coin is heads, training will use the last true value
            if i < self.training_params["initial_teacher_period"]:
                coin_heads = True
            else:
                p = sigmoid_decay(self.current_epoc, scale=self.training_params["teacher_sampling_decay"])
                coin_heads = coin_flip(p)

            if coin_heads:   # updates the training sample to include next true value                               
                sampled_X = torch.cat((                    
                        sampled_X.detach(), 
                        X[i+1, :, :].unsqueeze(0).detach()
                    ))    
            else:           # updates the training sample to include the last prediction                             
                new_features = X[i+1,:, self.output_size:].unsqueeze(0)

                last_pred_X = torch.cat((pred[-1,:,:].unsqueeze(0), new_features), dim=2)
                sampled_X = torch.cat((sampled_X.detach(), last_pred_X.detach()))

        loss = self.loss(Y[:-1,:, 0:self.output_size], pred)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X_i, Y_i, X, Y, group = batch
        X = X.permute(1,0,2)[1:, :, :]  # training data shifted left by 1.
        Y = Y.permute(1,0,2)            # training shifted right by 1.

        next_input = X
        all_pred = []

        for i in range(self.dataset_params["forecast_window"] - 1):
            pred = self.forward(next_input)

            if all_pred == []:
                all_pred = pred
            else:
                all_pred = torch.cat((all_pred, pred[-1,:,:].unsqueeze(0)))


            old_features = X[i + 1:, :, self.output_size:]                     # Size: [train_window - 1, 1, feature_size]
            new_feature = Y[i + 1, :, self.output_size:].unsqueeze(1)          # Size: [1, 1, feature_size]
            new_features = torch.cat((old_features, new_feature))          # Size: [train_window, 1, feature_size]
            
            next_input = torch.cat((X[i+1:, :, 0:self.output_size], pred[-1,:,:].unsqueeze(0)))
            next_input = torch.cat((next_input, new_features), dim = 2)
        true = torch.cat((X[1:,:,0:self.output_size], Y[:-1,:,0:self.output_size]))
        loss = self.loss(true, all_pred[:,:, 0:self.output_size])
        self.log('val_loss', loss, on_step=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200)
        return {'optimizer': optimizer } # 'lr_scheduler': scheduler, "monitor": self.loss }

    def forward(self, X: torch.Tensor):
        return NotImplementedError

####################################
## Model Implementations ###########
####################################

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

class LSTM(LitBase):
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

ALL = {
    "transformer": TransformerSeries,
    "lstm": LSTM,
    "baseline": Baseline,
}