# TransformerSeries

A general purpose implementation of the Transformer architecture for multivariate timeseries.

### Quick Start

To preprocess your csv file, run the comand below with the following:
- The path of your csv file. Please note with gcsfs installed this can be a GCS object you have access to.
- The name of the data column in the csv
- A string representation of a list of columns names containing all of columns that categorise a unqiue timeseries.

```
python src/preprocess.py 
    // --csv_file path
    // --date_column column_name
    // --group_columns "['list', 'of', 'column', 'names']"
```

If not date_column  or group_columns is provided, the default is "date" and None respectively.

Run the following command to train the model.

`make train`

Run the following command to run tensorboard

`make tensorboard`

## Overview

This repo implements a configurable train / evaluation loop for timeseries data. It is built to
operate with pandas dataframes.

The preprocess.py file preprocess a csv file by completing the following:
 - It ensures that the timeseries is in time ascending order.
 - It encodes a time position using a sin / cos encoding of the hour, day and month frequencies.
 - It sorts the column order of features. Order as follows: Targets / Raw Features / Engineered Features
 - It creates a test / train split and saves the files in DATA_DIR.

The TimeSeriresDataset class implements the following transformations to the data:
 - It min/max scales each timeseries to avoid target leakage.

## Training
The training loop utilises teacher forcing (with sampling) to improve the training process and 
avoid overfitting.

## Metrics
The metrics calculated in the evaluation loop can be configured in the configuration file. Evaluation
metrics and sample plots are all viewable within tensorboard.

## Models
The main focus of this repo is the exploration of Transformer architecture for timeseries analysis.
However, the following models are also included for benchmarking purposes:
 - A Baseline heuristic model
 - A vanila LSTM model with a dense final layer

## Configuration

- TO DO: Move the configuration from .env for a YAML file with models / datasets definable similar to docker-compose services.

## To Do

- Weights & Bias integration for hyperparameter tuning
- Pulumi Integration for model serving Infra as Code
- Grid.ai integration for easy training
- Look at borrowing from Hugginfaces model classes to leverage there tooling
