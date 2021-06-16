# TransformerSeries

A general purpose implementation of the Transformer architecture for multi-variate timeseries.

### Quick Start

To preprocess your csv file, run the comand below with the following:
- The path of your csv file
- The name of the data column in the csv
- A string representation of a list of columns names containing all of columns that categorise a unqiue timeseries.

```
python src/dataset.py --csv_file {path} 
    // --date_column {column_name} 
    // --group_columns {"["list", "of", "column", "names"]"}
```

If not date_column  or group_columns is provided, the default is "date" and None respectively.

Run the following command to train the model.

`make train`

Run the following command to run tensorboard

`make tensorboard`

## Overview

This repo implements a configurable train / evaluation loop for timeseries data. It is built to
operate with pandas dataframes.

The dataset.py file preprocess a csv file by completing the following:
 - It ensures that the timeseries is in time ascending order.
 - It encodes a time position using a sin / cos encoding of the hour, day and month frequencies.
 - It sorts the column order of features.
 - It creates a test / train split and saves the files in DATA_DIR.
 - It min/max scales all input timeseries on load to avoid target leakage.

 The training loop utilises teacher forcing (with sampling) to improve the training process and 
 avoid overfitting.

 The metrics calculated in the evaluation loop can be configured in the configuration file. Evaluation
 metrics and sample plots are all viewable within tensorboard.

The main focus of this repo is the exploration of Transformer architecture for timeseries analysis.
However, the following models are also included for benchmarking purposes:
 - A Baseline heuristic model
 - A vanila LSTM model with a dense final layer
