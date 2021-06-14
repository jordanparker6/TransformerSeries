"""Configuration File."""
import os
import sys
from pathlib import Path

# ~~~~~ PATHS ~~~~~
DATA_DIR: str = Path(os.getenv('DATA_DIR', './data'))
MODEL_DIR: str = Path(os.getenv('MODEL_DIR', './models'))


# ~~~~~ MODEL CONFIG ~~~~~
MODEL = {
    "num_layers": int(os.getenv("MODEL_TRANSFORMER_LAYERS", 3)),
    "attn_heads": int(os.getenv("MODEL_ATTN_HEADS", 8)),
    "dropout": float(os.getenv("MODEL_DROPOUT", 0))
}
