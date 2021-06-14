import os
import shutil
import argparse
import logging
from torch.utils.data import DataLoader
from pathlib import Path

import config
from dataset import TimeSeriesDataset
from train import run_training
from predict import predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] | %(name)s | %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
data_dir = config.DATA_DIR
model_dir = config.MODEL_DIR
predictions_dir = config.MODEL_DIR.joinpath("predictions")
logs_dir = config.MODEL_DIR.joinpath("logs")

def cleanup(_dir: Path):
    if os.path.exists(_dir): 
        shutil.rmtree(_dir)
    os.mkdir(_dir)

def main(
    epochs: int = 1000,
    k: int = 60,
    batch_size: int = 1,
    training_length: int = 30 * 3,
    forecast_window: int = 30,
    initial_teacher_period: int = 30,
    device: str = "cpu"
):
    """Method to bring together clean-up, data-loading, pre-processing and training"""
    list(map(cleanup, [model_dir, predictions_dir, logs_dir]))

    train_dataset = TimeSeriesDataset(
            csv_file = data_dir.joinpath("train.csv"),
            targets=["Open", "High", "Low", "Close"],
            training_length = training_length, 
            forecast_window = forecast_window
        )
    test_dataset = TimeSeriesDataset(
            csv_file = data_dir.joinpath("test.csv"),
            targets=["Open", "High", "Low", "Close"],
            training_length = training_length,
            forecast_window = forecast_window
        )
    best_model = run_training(
            data=train_dataset, 
            epochs=epochs, 
            attn_heads=len(train_dataset.features),
            k=k, 
            initial_teacher_period=initial_teacher_period,
            model_dir=model_dir, 
            device=device
        )
    predict(
            data=test_dataset, 
            model_path=best_model,
            attn_heads=len(train_dataset.features),
            forecast_window=forecast_window, 
            predictions_dir=predictions_dir,
            device=device
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--k", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(
        epochs=args.epochs,
        k=args.k,
        batch_size=args.batch_size,
        device=args.device,
    )