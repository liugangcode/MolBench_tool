# MolBench: A Molecular Representation Benchmarking Toolkit

MolBench is a toolkit for evaluating molecular representations (e.g., cell morphology features) across multiple chemical prediction tasks [from InfoAlign](https://openreview.net/forum?id=BbZy8nI1si).

## Supported Datasets

- **ChEMBL2k**: Multi-task classification dataset with 41 tasks from ChEMBL bioactivity data
- **Broad6k**: Multi-task classification dataset with 32 tasks from Broad Institute screening data  
- **Biogen3k**: Multi-task regression dataset with 6 ADME property prediction tasks
- **ToxCast**: Large-scale multi-task classification dataset with 617 toxicity prediction tasks

## Supported Models

The toolkit currently supports the following machine learning models:

- **Random Forest (RF)** - Implemented using scikit-learn
- **Gaussian Process (GP)** - Implemented using scikit-learn  
- **Multi-Layer Perceptron (MLP)** - Implemented using PyTorch

## Core Functions

MolBench provides the following key functions:

- **load_task**: Downloads a task (e.g., ChEMBL2k) and loads the task-related information
- **match_task_data**: Matches molecular structures in cell morphology data with those in the task
- **train_predictor**: Trains a classifier or regressor on top of the input features using one of the supported models
- **evaluate**: Returns the AUC evaluation for each property in the task and returns the overall predictions
- **save_results/load_results/plot_results**: Functions to save, load, and create bar plots for each task

## Usage

An example use case can be found in the `examples` directory:
- Use `1_download_feature.py` to download cell morphology features
- Run the benchmarking with either `run.sh` or `2_test_bench.py`
