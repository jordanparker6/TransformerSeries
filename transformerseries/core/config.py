"""Configuration File."""
import os
import sys
from pathlib import Path
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ~~~~~ PATHS ~~~~~
DATA_DIR: str = Path(os.getenv('DATA_DIR', './train/data'))
MODEL_DIR: str = Path(os.getenv('MODEL_DIR', './train/models'))
SERVE_DIR: str = Path(os.getenv('MODEL_DIR', './serve'))

# ~~~~~ DATASET CONFIG ~~~~~
DATASET = {
    "targets": ["y1", "y2"],
    "training_length": 7, #168
    "forecast_window": 3, #24,
    "batch_size": 1,
    "num_workers": 4
}

# ~~~~~ MODEL CONFIG ~~~~~
MODEL = {
    "model": "transformer",
    "task": "regression",
    "epochs": int(os.getenv("MODEL_EPOCHS", 1000)),
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
    "initial_teacher_period": int(os.getenv("MODEL_INIT_TEACHER_PERIOD", 30)),
    "teacher_sampling_decay": float(os.getenv("MODEL_TEACHER_SAMPLE_DECAY", 60)),
    "loss": "MSE",
    "metrics": ["MAE", "MSE"]
}

# ~~~~~ GLOBAL METRICS OBJECT ~~~~~
METRICS = {
    "MSE": torch.nn.MSELoss(),
    "MAE": torch.nn.L1Loss(),
    "MAPE": lambda y, yhat: ((y - yhat).abs() / (y.abs() + 1e-8)).mean()
}