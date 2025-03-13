# MolBench: A Molecular Representation Benchmarking Toolkit

MolBench is a toolkit for evaluating molecular representations (e.g., cell morphology features) across multiple chemical prediction tasks [from InfoAlign](https://openreview.net/forum?id=BbZy8nI1si).

## Supported Datasets

- **ChEMBL2k**: Multi-task classification dataset with 41 tasks from ChEMBL bioactivity data
- **Broad6k**: Multi-task classification dataset with 32 tasks from Broad Institute screening data  
- **Biogen3k**: Multi-task regression dataset with 6 ADME property prediction tasks
- **ToxCast**: Large-scale multi-task classification dataset with 617 toxicity prediction tasks

## Supported Classifier

The toolkit currently supports the following machine learning models:

- Random Forest (RF) - Implemented using scikit-learn
- Gaussian Process (GP) - Implemented using scikit-learn  
- Multi-Layer Perceptron (MLP) - Implemented using PyTorch

## Usage
An example use case can be found in the `examples` directory, using `run.sh` or `2_test_bench.py`. One may download the cell morphology features using `1_download_feature.py`.
