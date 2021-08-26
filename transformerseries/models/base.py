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

class BaseModel(LightningModule):
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