import logging
import torch
import joblib
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from model import TransformerSeries
from plot import plot_prediction

logger = logging.getLogger(__name__)
log_dir = config.MODEL_DIR.joinpath("logs")
writer = SummaryWriter(config.MODEL_DIR.joinpath("logs"), comment="test")

def predict(
        data: Dataset,
        model_path: Path,
        attn_heads: int,
        forecast_window: int,
        device: str, 
        predictions_dir: str, 
    ):
    device = torch.device(device)
    features = data.features
    raw_features = data.raw_features
    targets = data.targets
    date_index = data.dates

    dataloader = DataLoader(data, batch_size=1, shuffle=True)
    model = TransformerSeries(
            feature_size=len(features),
            output_size=len(targets),
            attn_heads=attn_heads
        ).double().to(device)
    model.load_state_dict(torch.load(model_path))
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        model.eval()

        all_val_loss = []
        for plot in range(25):
            val_loss = 0
            for X_i, Y_i, X, Y, group in dataloader:
        
                #X = X.permute(1,0,2).double().to(device)[1:, :, :]
                X = X.permute(1,0,2).double().to(device)[1:, :, :]
                Y = Y.permute(1,0,2).double().to(device)

                next_input = X
                all_pred = []

                for i in range(forecast_window - 1):
                    pred = model(next_input, device)

                    if all_pred == []:
                        all_pred = pred
                    else:
                        all_pred = torch.cat((all_pred, pred[-1,:,:].unsqueeze(0)))

                    old_features = X[i + 1:, :, len(targets):]                     # Size: [train_window - 1, 1, feature_size]
                    new_feature = Y[i + 1, :, len(targets):].unsqueeze(1)          # Size: [1, 1, feature_size]
                    new_features = torch.cat((old_features, new_feature))          # Size: [train_window, 1, feature_size]
                    
                    next_input = torch.cat((X[i+1:, :, 0:len(targets)], pred[-1,:,:].unsqueeze(0)))
                    next_input = torch.cat((next_input, new_features), dim = 2)

                true = torch.cat((X[1:,:,0:len(targets)], Y[:-1,:,0:len(targets)]))
                loss = criterion(true, all_pred[:,:, 0:len(targets)])
                val_loss += loss
            
            val_loss = val_loss / 10
            all_val_loss.append(val_loss)
            logger.info(f"Sample: {plot}, Validation loss: {val_loss}")
            writer.add_scalar("validation_loss", val_loss, plot)
            
            if plot % 5 == 0:
                scalar = joblib.load('scalar_item.joblib')
                X = data.inverse_transform(X, scalar)
                Y = data.inverse_transform(Y, scalar)
                all_pred = data.inverse_transform(all_pred, scalar)
                X_dates = date_index[X_i.tolist()[0]].tolist()[1:]
                Y_dates = date_index[Y_i.tolist()[0]].tolist()

                for i, target in enumerate(targets):
                    writer.add_figure(
                            f"test_plot_'{target}'@sample-{plot}", 
                            plot_prediction(X[:, i], Y[:, i], all_pred[:, i], X_dates, Y_dates), 
                            plot
                        )

        all_val_loss = torch.cat([x.unsqueeze(0) for x in all_val_loss])
        logger.info(f"Average Validation Loss: {all_val_loss.mean()}")
