# MolBench: A Molecular Representation Benchmarking Toolkit

MolBench is a toolkit for evaluating and benchmarking molecular representations across multiple chemical prediction tasks.

## Supported Datasets

- **ChEMBL2k**: Multi-task classification dataset with 41 tasks from ChEMBL bioactivity data
- **Broad6k**: Multi-task classification dataset with 32 tasks from Broad Institute screening data  
- **Biogen3k**: Multi-task regression dataset with 6 ADME property prediction tasks
- **ToxCast**: Large-scale multi-task classification dataset with 617 toxicity prediction tasks

## Supported Models

The toolkit currently supports the following machine learning models:

- Random Forest (RF) - Implemented using scikit-learn
- Gaussian Process (GP) - Implemented using scikit-learn  
- Multi-Layer Perceptron (MLP) - Implemented using PyTorch
