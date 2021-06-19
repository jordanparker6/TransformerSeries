from typing import List
import os
import shutil
import logging
from pathlib import Path

import config
import train
from dataset import TimeSeriesDataset
from evaluate import evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] | %(name)s | %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")

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
    list(map(cleanup, [
        model_dir, 
        model_dir.joinpath("predictions"), 
        model_dir.joinpath("logs")
    ]))

    train_dataset = TimeSeriesDataset(
            csv_file=data_dir.joinpath("train.csv"),
            targets=targets,
            training_length = training_length, 
            forecast_window = forecast_window
        )
    test_dataset = TimeSeriesDataset(
            csv_file=data_dir.joinpath("test.csv"),
            targets=targets,
            training_length=training_length,
            forecast_window=forecast_window
        )
    best_model_path = train.run_teacher_forcing_training(
            model_name=model_name,
            data=train_dataset, 
            epochs=epochs,
            k=k, 
            initial_teacher_period=initial_teacher_period,
            model_dir=model_dir
        )
    evaluate(
            model_name=model_name,
            data=test_dataset, 
            model_path=best_model_path,
            forecast_window=forecast_window,
            model_dir=model_dir
        )
    shutil.copyfile(best_model_path, serve_dir.joinpath(f"{model_name}_model.pth"))

if __name__ == "__main__":
    for model in ["baseline", "transformer", "lstm"]:
        main(model_name=model, model_dir=config.MODEL_DIR.joinpath(model))