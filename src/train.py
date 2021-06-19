import logging
import math
import random
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

import config
import models
from plot import plot_teacher_forcing

logger = logging.getLogger(__name__)
log_dir = config.MODEL_DIR.joinpath("logs")

#########################
## Utility Functions ####
#########################

def coin_flip(p):
    return True if random.random() < p else False

def sigmoid_decay(x: float, scale: float):
    return scale / (scale + math.exp(x / scale))

#########################
## Training Function ####
#########################

def run_teacher_forcing_training(
        model_name: str,
        data: Dataset, 
        epochs: int,
        k: float,
        initial_teacher_period: int,
        model_dir: Path
    ) -> Path:
    """
    Runs training with sampling over teacher forcing.

    The training objective is to predict the value of the next timestep.

    Teacher forcing is used to speed up the training time. Teacher forcing
    injects the last true value into the input data to correct the models 
    forecast. This technique has some drawbacks. It can lead to poor 
    performance on test data for unseen ranges. To overcome this,
    a sampling technique is implemented to randomly jump between teacher forcing and
    ordinary trianing. The probability of transitioning is set to decay over the epochs,
    leaving less teacher invovlement in later epochs. 

    Args:
        data (Dataset): A PyTorch dataset class.
        epoch (int): The number of training iterations.
        k (float)): The scale parameter for the sampling probability decay.
        path_to_save_model (str): [description]
        path_to_save_loss (str): [description]
        path_to_save_predictions (str): [description]
    """
    device = torch.device(config.DEVICE)
    features = data.features
    raw_features = data.raw_features
    targets = data.targets
    date_index = data.dates

    model = models.ALL[model_name]
    model = model(feature_size=len(features), output_size=len(targets)).double().to(device)
    dataloader = DataLoader(data, batch_size=1, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200)
    criterion = config.METRICS[config.MODEL["loss"]]
    best_model = ""
    min_train_loss = float('inf')
    writer = SummaryWriter(model_dir.joinpath("logs"), comment="training")

    for epoch in range(epochs + 1):
        train_loss = 0
        model.train()
        for X_i, Y_i, _X, _Y, group in dataloader:
            optimizer.zero_grad()
            
            # Dataloader --> Model
            # Permute: [batch, input_length, feature] --> [input_length, batch, feature]
            X = _X.permute(1,0,2).double().to(device)[:-1,:,:]  # training data shifted left by 1.
            Y = _X.permute(1,0,2).double().to(device)[1:,:,:]   # training shifted right by 1.

            # init training sample as the first true sample  
            sampled_X = X[:1, :, :]   
                                
            for i in range(len(Y)-1):
                pred = model(sampled_X)

                # if coin is heads, training will use the last true value
                if i < initial_teacher_period:
                    coin_heads = True
                else:
                    p = sigmoid_decay(epoch, scale=k)
                    coin_heads = coin_flip(p)

                if coin_heads:   # updates the training sample to include next true value                               
                    sampled_X = torch.cat((                    
                            sampled_X.detach(), 
                            X[i+1, :, :].unsqueeze(0).detach()
                        ))    
                else:           # updates the training sample to include the last prediction                             
                    new_features = X[i+1,:, len(targets):].unsqueeze(0)
            
                    last_pred_X = torch.cat((pred[-1,:,:].unsqueeze(0), new_features), dim=2)
                    sampled_X = torch.cat((sampled_X.detach(), last_pred_X.detach()))

            loss = criterion(Y[:-1,:, 0:len(targets)], pred)
            loss.backward()
            optimizer.step()
            scheduler.step(loss.detach().item())
            train_loss += loss.detach().item()

        writer.add_scalar(f"{model_name}_train_loss", train_loss, epoch)

        # Save the best model
        if train_loss < min_train_loss:
            best_model = model_dir.joinpath(f"best_train_{epoch}.pth")
            optimizer_state = model_dir.joinpath(f"optimizer_{epoch}.pth")
            torch.save(model.state_dict(), best_model)
            torch.save(optimizer.state_dict(), optimizer_state)
            min_train_loss = train_loss
        
        if epoch % 10 == 0: # log training
            logger.info(f"{model_name} | Epoch: {epoch}, Training loss: {train_loss}")

        if epoch % 100 == 0:
            scalar = joblib.load('scalar_item.joblib')
            sampled_X = data.inverse_transform(sampled_X, scalar)
            X = data.inverse_transform(X, scalar)
            Y = data.inverse_transform(Y, scalar)
            
            print(pred.shape)
            print(pred)

            pred = data.inverse_transform(pred, scalar)
            X_dates = date_index[X_i.tolist()[0]][:-1].tolist()
            Y_dates = date_index[X_i.tolist()[0]][1:].tolist()
            for i, target in enumerate(targets):
                print(target)
                print(i)
                print(X.shape)
                print(X[:, i])
                print(pred.shape)
                print(pred[:, i])
                writer.add_figure(
                        f"{model_name}_train_plot_'{target}'@epoch-{epoch}", 
                        plot_teacher_forcing(X_dates, X[:, i], sampled_X[:, i], pred[:, i], epoch), 
                        epoch
                    )
 
    return best_model