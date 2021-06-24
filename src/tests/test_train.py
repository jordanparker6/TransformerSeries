import os, sys
import pytest
from pathlib import Path

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

import train
from dataset import TimeSeriesDataset
import config

@pytest.fixture
def dataset():
    return TimeSeriesDataset(
        csv_file= Path("unittest/train.csv"),
        targets=["y1", "y2"],
        training_length=10,
        forecast_window=3
    )

def test_train(dataset):
    train.run_teacher_forcing_training(
        model_name="transformer",
        data=dataset,
        epochs=1000,
        k=60,
        initial_teacher_period=2,
        model_dir=Path("unittest/model")
    )