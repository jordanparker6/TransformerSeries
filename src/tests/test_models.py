import os, sys
import pytest
import torch
from torchtest import assert_vars_change

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

import models

BATCH = (torch.ones(10, 1, 4).double(), torch.ones(10, 1, 1).double())
LOSS = torch.nn.MSELoss()
OPTIMIZER = torch.optim.Adam
DEVICE = "cpu"

def print_named_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

###############################################
## Test Model Variables Change on Backprop ####
###############################################

def test_baseline_vars_change():
    model = models.ALL["baseline"](
        feature_size=BATCH[0].shape[2], 
        output_size=BATCH[1].shape[2]
    ).double()
    print_named_params(model)
    assert_vars_change(
        model=model,
        loss_fn=LOSS,
        optim=OPTIMIZER(model.parameters()),
        batch=BATCH,
        device=DEVICE
    )

def test_lstm_vars_change():
    model = models.ALL["lstm"](
        feature_size=BATCH[0].shape[2], 
        output_size=BATCH[1].shape[2]
    ).double()
    print_named_params(model)
    assert_vars_change(
        model=model,
        loss_fn=LOSS,
        optim=OPTIMIZER(model.parameters()),
        batch=BATCH,
        device=DEVICE
    )

def test_transformer_vars_change():
    model = models.ALL["transformer"](
        feature_size=BATCH[0].shape[2], 
        output_size=BATCH[1].shape[2]
    ).double()
    print_named_params(model)
    assert_vars_change(
        model=model,
        loss_fn=LOSS,
        optim=OPTIMIZER(model.parameters()),
        batch=BATCH,
        device=DEVICE
    )