"""Configuration File."""
import os
import sys
from pathlib import Path

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
    "num_layers": int(os.getenv("MODEL_TRANSFORMER_LAYERS", 3)),
    "attn_heads": int(os.getenv("MODEL_ATTN_HEADS", 1)),
    "dropout": float(os.getenv("MODEL_DROPOUT", 0.1)),
    "epochs": int(os.getenv("MODEL_EPOCHS", 5000)),
    "initial_teacher_period": int(os.getenv("MODEL_INIT_TEACHER_PERIOD", 30)),
    "teacher_sampling_decay": float(os.getenv("MODEL_TEACHER_SAMPLE_DECAY", 60))
}
