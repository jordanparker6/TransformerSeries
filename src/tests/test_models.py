import os, sys
import pytest
import torch
from torchtest import assert_vars_change
from pathlib import Path

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

import models
import config
from dataset import TimeSeriesDataset

BATCH = (torch.ones(10, 1, 4).double(), torch.ones(10, 1, 1).double())
LOSS = torch.nn.MSELoss()
OPTIMIZER = torch.optim.Adam
DEVICE = config.DEVICE

def print_named_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

@pytest.fixture
def dataset():
    return TimeSeriesDataset(
        csv_file= Path("unittest/train.csv"),
        targets=["y1", "y2"],
        training_length=10,
        forecast_window=3
    )

###############################################
## Test Model Variables Change on Backprop ####
###############################################

def test_baseline_vars_change(dataset):
    model = models.ALL["baseline"](dataset).double().to(DEVICE)
    print_named_params(model)
    assert_vars_change(
        model=model,
        loss_fn=LOSS,
        optim=OPTIMIZER(model.parameters()),
        batch=dataset,
        device=DEVICE
    )

def test_lstm_vars_change(dataset):
    model = models.ALL["lstm"](dataset).double().to(DEVICE)
    print_named_params(model)
    assert_vars_change(
        model=model,
        loss_fn=LOSS,
        optim=OPTIMIZER(model.parameters()),
        batch=dataset,
        device=DEVICE
    )

def test_transformer_vars_change(dataset):
    model = models.ALL["transformer"](dataset).double().to(DEVICE)
    print_named_params(model)
    assert_vars_change(
        model=model,
        loss_fn=LOSS,
        optim=OPTIMIZER(model.parameters()),
        batch=dataset,
        device=DEVICE
    )