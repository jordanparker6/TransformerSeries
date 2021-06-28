from typing import List
import os
import shutil
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, GPUStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import config
from dataset import TimeSeriesDataset
import models

pl.seed_everything(42, workers=True)

def cleanup(_dir: Path):
    if os.path.exists(_dir): 
        shutil.rmtree(_dir)
    os.mkdir(_dir)

def main(
    model_name: str = config.MODEL["model"],
    targets: List[str] = config.DATASET["targets"],
    batch_size: int = config.DATASET["batch_size"],
    training_length: int = config.DATASET["training_length"],
    forecast_window: int = config.DATASET["forecast_window"],
    epochs: int = config.MODEL["epochs"],
    initial_teacher_period: int = config.MODEL["initial_teacher_period"],
    k: int = config.MODEL["teacher_sampling_decay"],
    model_dir: Path = config.MODEL_DIR,
    data_dir: Path = config.DATA_DIR,
    serve_dir: Path = config.SERVE_DIR
):
    """Method to bring together clean-up, data-loading, pre-processing and training"""

    print("""
 _____                     __                                ____            _           
|_   _| __ __ _ _ __  ___ / _| ___  _ __ _ __ ___   ___ _ __/ ___|  ___ _ __(_) ___  ___ 
  | || '__/ _` | '_ \/ __| |_ / _ \| '__| '_ ` _ \ / _ \ '__\___ \ / _ \ '__| |/ _ \/ __|
  | || | | (_| | | | \__ \  _| (_) | |  | | | | | |  __/ |   ___) |  __/ |  | |  __/\__ \
  |_||_|  \__,_|_| |_|___/_|  \___/|_|  |_| |_| |_|\___|_|  |____/ \___|_|  |_|\___||___/
    """)

    list(map(cleanup, [
        model_dir, 
        model_dir.joinpath("predictions"), 
        model_dir.joinpath("logs")
    ]))

    # DEFINE: Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            stopping_threshold=1e-4,
            divergence_threshold=9.0,
            check_finite=True
        ),
        ModelCheckpoint(
            monitor='val_loss',
            dirpath=config.MODEL_DIR,
            filename=model_name + '-epoch{epoch:02d}-val_loss{val_loss:.2f}',
            auto_insert_metric_name=False
        )
    ]
    if torch.cuda.is_available():
        callbacks +=  [GPUStatsMonitor()]

    # BUILD: Dataset & DataLoaders
    train_dataset = TimeSeriesDataset(
            csv_file=data_dir.joinpath("train.csv"),
            targets=targets,
            training_length = training_length, 
            forecast_window = forecast_window
        )
    val_dataset = TimeSeriesDataset(
            csv_file=data_dir.joinpath("test.csv"),
            targets=targets,
            training_length=training_length,
            forecast_window=forecast_window
        )
    train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config.DATASET["batch_size"],
            num_workers=config.DATASET["num_workers"],
            pin_memory=True
        )
    val_dataloader = DataLoader(
            val_dataset, 
            batch_size=config.DATASET["batch_size"],
            num_workers=config.DATASET["num_workers"],
            pin_memory=True
        )
    
    # BUILD: Model
    model = models.Baseline(train_dataset).double()

    # RUN: Training / Validation / Testing
    #logger = TensorBoardLogger(model_dir.joinpath("logs"))
    logger = TensorBoardLogger(save_dir=model_dir.joinpath("logs"))
    trainer = pl.Trainer(
            gpus=torch.cuda.device_count(), 
            callbacks=callbacks,
            logger=logger,
            max_epochs=epochs,
            stochastic_weight_avg=True,
            gradient_clip_val=0.1,
            profiler='pytorch'
        )
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(test_dataloaders=val_dataloader)

    #shutil.copyfile(best_model_path, serve_dir.joinpath(f"{model_name}_model.pth"))

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()