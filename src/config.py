"""Configuration File."""
import os
import sys
from pathlib import Path
import torch

DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

# ~~~~~ PATHS ~~~~~
DATA_DIR: str = Path(os.getenv('DATA_DIR', './train/data'))
MODEL_DIR: str = Path(os.getenv('MODEL_DIR', './train/models'))
SERVE_DIR: str = Path(os.getenv('MODEL_DIR', './serve'))

# ~~~~~ DATASET CONFIG ~~~~~
DATASET = {
    "targets": ["Open", "High", "Low", "Close"],
    "training_length": 90,
    "forecast_window": 30,
    "batch_size": 1
}

# ~~~~~ MODEL CONFIG ~~~~~
MODEL = {
    "model": "transformer",
    "task": "regression",
    "transformer": {
        "num_layers": int(os.getenv("MODEL_TRANSFORMER_LAYERS", 3)),
        "attn_heads": int(os.getenv("MODEL_TRANSFORMER_ATTN_HEADS", 1)),
        "num_layers": int(os.getenv("MODEL_TRANSFORMER_LAYERS", 3))
    },
    "lstm": {
        "hidden_states": int(os.getenv("MODEL_LSTM_HIDDEN_STATES", 100)),
        "layers": int(os.getenv("MODEL_LSTM_LAYERS", 2)),
    },
    "dropout": float(os.getenv("MODEL_DROPOUT", 0.1)),
    "epochs": int(os.getenv("MODEL_EPOCHS", 100)),
    "initial_teacher_period": int(os.getenv("MODEL_INIT_TEACHER_PERIOD", 30)),
    "teacher_sampling_decay": float(os.getenv("MODEL_TEACHER_SAMPLE_DECAY", 60)),
    "loss": "MSE",
    "metrics": ["MAPE", "MSE"]
}

# ~~~~~ GLOBAL METRICS OBJECT ~~~~~
METRICS = {
    "MSE": torch.nn.MSELoss(),
    "MAE": torch.nn.L1Loss(),
    "MAPE": lambda y, yhat: ((y - yhat).abs() / (y.abs() + 1e-8)).mean()
}