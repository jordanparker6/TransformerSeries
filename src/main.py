from typing import List
import os
import shutil
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, GPUStatsMonitor
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss

import config

#pl.seed_everything(42, workers=True)

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
    ///////////////////////////////////////////////////////////
    //  TransformerSeries Forecasting: Starting Training //////
    ///////////////////////////////////////////////////////////
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
    data = pd.read_csv(data_dir.joinpath("data.csv"), parse_dates=["timestamp"], index_col="timestamp").reset_index(drop=True)
    training_cutoff = data.time_idx[int(len(data.time_idx.unique()) * 0.7)]

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=config.DATASET["targets"],
        group_ids=["group_id"],
        min_encoder_length=config.DATASET["training_length"],
        max_encoder_length=config.DATASET["training_length"],
        min_prediction_length=1,
        max_prediction_length=config.DATASET["forecast_window"],
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=[],
        time_varying_known_reals=["time_idx", "sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month", "cos_month", "year"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[],
        target_normalizer=GroupNormalizer(
            groups=["group_id"], transformation="softplus"
        ),  # use softplus and normalize by group
        #add_relative_time_idx=True,         #adds time index as feature
        add_target_scales=True,             #adds scaling factors as features
        add_encoder_length=True,            #adds encoder length to list of static reals
    )
    batch_size = 1
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=config.DATASET["num_workers"])
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=config.DATASET["num_workers"])

    # BUILD: Dataset & DataLoaders
    actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
    baseline_predictions = Baseline().predict(val_dataloader)

    print("// BASELINE PERFORMANCE: MAE")
    print((actuals - baseline_predictions).abs().mean().item())

    # BUILD: Model
    print("// BUILDING MODEL")
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=7,  # 7 quantiles by default
        loss=QuantileLoss(),
        log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        reduce_on_plateau_patience=4,
    )

    # RUN: Training / Validation / Testing
    logger = TensorBoardLogger(model_dir.joinpath("logs"))
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(), 
        callbacks=callbacks,
        max_epochs=epochs,
        stochastic_weight_avg=True,
        gradient_clip_val=0.1,
        profiler='pytorch'
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(test_dataloaders=val_dataloader)

    best_model_path = trainer.checkpoint_callback.best_model_path
    print(best_model_path)
    best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    raw_predictions, x = best_model.predict(val_dataloader, mode="raw", return_x=True)
    for idx in range(10):  # plot 10 examples
        best_model.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True);

    #shutil.copyfile(best_model_path, serve_dir.joinpath(f"{model_name}_model.pth"))

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()


"""
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
    train_loader = DataLoader(
            train_dataset, 
            batch_size=config.DATASET["batch_size"],
            num_workers=config.DATASET["num_workers"],
            pin_memory=True
        )
    val_loader = DataLoader(
            val_dataset, 
            batch_size=config.DATASET["batch_size"],
            num_workers=config.DATASET["num_workers"],
            pin_memory=True
        )
"""