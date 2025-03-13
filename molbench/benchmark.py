import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import requests
import gzip
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from .data_utils import ensure_inchikey_column
from .split_utils import append_scaffold_split_column
from .feature_utils import get_fingerprint_features
from .metric_utils import compute_metric
from warnings import warn
from .model_utils import MLPPredictor

class MolBench:
    def __init__(self):
        self.task_urls = {
            'chembl2k': 'https://github.com/liugangcode/InfoAlign/raw/refs/heads/main/raw_data/chembl2k/raw/assays.csv.gz',
            'biogen3k': 'https://github.com/liugangcode/InfoAlign/raw/refs/heads/main/raw_data/biogenadme/raw/assays.csv.gz',
            'broad6k': 'https://github.com/liugangcode/InfoAlign/raw/refs/heads/main/raw_data/broad6k/raw/assays.csv.gz',
            'toxcast': 'https://github.com/liugangcode/InfoAlign/raw/refs/heads/main/raw_data/moltoxcast/raw/assays.csv.gz'
        }
        self.task_configs = {
            'chembl2k': {'num_tasks': 41, 'start_column': 4, 'task_type': 'classification'},
            'broad6k': {'num_tasks': 32, 'start_column': 2, 'task_type': 'classification'},
            'biogen3k': {'num_tasks': 6, 'start_column': 4, 'task_type': 'regression'},
            'toxcast': {'num_tasks': 617, 'start_column': 2, 'task_type': 'classification'}
        }
        self.task_type = None
        self.task_name = None
        self.feature_cols = None
        self.task_columns = None
        self.num_tasks = None
        self.task_df = None
        self.task_merged_df = None
        self.need_fingerprint_model = None
        self.evaluation_results = None
        self.model_type = None
    
    def get_available_tasks(self):
        return list(self.task_configs.keys())
    
    def _download_data(self, url, data_path):
        """Download data from URL and save to specified path."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(data_path, 'wb') as f:
            f.write(response.content)
    
    def load_task_data(self, task_name, data_path='raw_data'):
        """Load and process task data.
        
        Args:
            task_name (str): Name of the task ('chembl2k', 'broad6k', 'toxcast', 'biogen3k')
            data_path (str): Directory path to store/load data
            
        Returns:
            pandas.DataFrame: Processed dataset
        """
        if task_name not in self.task_configs:
            raise ValueError(f"Unsupported task: {task_name}. Supported tasks are: {list(self.task_configs.keys())}")
        
        # Create data directory if it doesn't exist
        os.makedirs(data_path, exist_ok=True)
        
        # Construct file path
        file_path = os.path.join(data_path, f"{task_name}.csv.gz")
        
        # Download data if it doesn't exist
        if not os.path.exists(file_path):
            print(f"Downloading {task_name} dataset...")
            self._download_data(self.task_urls[task_name], file_path)
        
        # Read the gzipped CSV file
        try:
            data = pd.read_csv(file_path, compression='gzip')
        except Exception as e:
            raise Exception(f"Error reading {task_name} dataset: {str(e)}")
        
        # Get task configuration
        config = self.task_configs[task_name]
        start_col = config['start_column']
        num_tasks = config['num_tasks']
        task_type = config['task_type']
        
        # Validate that we have enough columns
        if len(data.columns) < start_col + num_tasks:
            raise ValueError(f"Dataset has insufficient columns. Expected at least {start_col + num_tasks}, but got {len(data.columns)}")

        data = append_scaffold_split_column(data)
        data = ensure_inchikey_column(data, 'smiles')
        
        self.task_name = task_name
        self.task_type = task_type
        self.task_columns = data.columns[start_col:start_col+num_tasks]
        self.task_df = data
        self.num_tasks = num_tasks

        self.feature_cols = None
        self.task_merged_df = None
        self.need_fingerprint_model = None
        self.evaluation_results = None
        self.model_type = None

        return self.task_df
    
    def match_task_data(self, data, merge_key='inchikey', feature_prefix='X_'):
        """
        Adds feature columns from the input dataframe to the task dataframe.
        
        Args:
            data (pd.DataFrame): Input dataframe with molecular features
            merge_key (str): Column name to use for merging (default: 'inchikey')
            feature_prefix (str): Prefix used to identify feature columns (default: 'X_')
            
        Returns:
            pd.DataFrame: Task dataframe with added feature columns
        """
        self.feature_cols = [col for col in data.columns if col.startswith(feature_prefix)]
        task_df = self.task_df        
        if merge_key not in data.columns:
            raise ValueError(f"Merge key '{merge_key}' not found in input dataframe")
        
        if merge_key not in task_df.columns:
            raise ValueError(f"Merge key '{merge_key}' not found in task dataframe and cannot be derived")
        
        # Identify feature columns in the input dataframe
        if self.feature_cols is None:
            raise ValueError(f"No feature columns with prefix '{feature_prefix}' found in input dataframe")
        
        print(f"Found {len(self.feature_cols)} feature columns with prefix '{feature_prefix}'")
        
        # Create a subset of the input dataframe with only the merge key and feature columns
        data_subset = data[[merge_key] + self.feature_cols].copy()
        
        # Merge the dataframes - adding features to the task dataframe
        print(f"Merging feature data into task data using key: {merge_key}")
        merged_df = pd.merge(task_df, data_subset, on=merge_key, how='left')
        
        # Check for missing feature data
        missing_features_mask = merged_df[self.feature_cols].isna().any(axis=1)
        missing_features_count = missing_features_mask.sum()
        
        if missing_features_count > 0:
            print(f"Warning: {missing_features_count} rows ({missing_features_count/len(merged_df):.2%}) in the task data have missing feature values")
            
            # Check distribution of missing features by split
            if 'split' in merged_df.columns:
                print("\nDistribution of missing features by split:")
                split_stats = merged_df[missing_features_mask].groupby('split').size()
                split_totals = merged_df.groupby('split').size()
                
                for split in split_totals.index:
                    missing_in_split = split_stats.get(split, 0)
                    total_in_split = split_totals.get(split)
                    print(f"  {split}: {missing_in_split}/{total_in_split} ({missing_in_split/total_in_split:.2%}) rows have missing features")
        
        self.task_merged_df = merged_df
        return merged_df
    
    def train_predictor(self, model_type='RF', verbose=False):
        """Trains predictors for each task using evaluated features when available,
        with auxiliary fingerprint-based models as backup.
        
        Args:
            model_type (str): Type of model to use ('RF', 'GP', or 'MLP')
            verbose (bool): Whether to print progress information and show progress bars
        
        Returns:
            dict: Dictionary containing trained models for each task
        """
        
        self.model_type = model_type
        # Check if processed data is available
        if self.task_merged_df is None:
            raise ValueError("Task data is not merged with feature data. Process data first with match_task_data.")
            
        # Initialize model dictionaries
        self.eval_feature_models = {}  # Models trained on evaluated features
        self.fingerprint_models = {}   # Auxiliary models trained on fingerprints
        
        # Get data splits
        train_df = self.task_merged_df[self.task_merged_df['split'] == 'train']
        valid_df = self.task_merged_df[self.task_merged_df['split'] == 'valid']
        test_df = self.task_merged_df[self.task_merged_df['split'] == 'test']
        
        if verbose:
            print(f"Training data: {len(train_df)} samples")
            print(f"Validation data: {len(valid_df)} samples")
            print(f"Test data: {len(test_df)} samples")
        
        has_eval_features = self.feature_cols is not None and len(self.feature_cols) > 0
        if not has_eval_features:
            raise ValueError("No features available.")
        
        if verbose:
            print(f"Using {len(self.feature_cols)} evaluated feature columns when available")
        # Extract evaluated features
        train_eval_features = train_df[self.feature_cols].values
        valid_eval_features = valid_df[self.feature_cols].values
        test_eval_features = test_df[self.feature_cols].values
        
        # Store evaluated features for later use
        self.train_eval_features = train_eval_features
        self.valid_eval_features = valid_eval_features
        self.test_eval_features = test_eval_features
        # need_fingerprint_model = np.isnan(valid_eval_features).any() or np.isnan(test_eval_features).any()
        need_fingerprint_model = True
        self.need_fingerprint_model = need_fingerprint_model

        # Handle MLP separately as it can be multitask
        if model_type == 'MLP':
            if verbose:
                print("Training MLP model for multitask learning")
            
            train_labels = train_df[self.task_columns].values            
            self.mlp_model = MLPPredictor(in_features=train_eval_features.shape[1], out_features=train_labels.shape[1], task_type=self.task_type)
            self.mlp_model.fit(train_eval_features, train_labels, verbose=verbose)                
            for task_col in self.task_columns:
                self.eval_feature_models[task_col] = self.mlp_model

            if need_fingerprint_model:
                if verbose:
                    print("Training auxiliary fingerprint-based MLP model")
                self.train_mol_fps = np.array([get_fingerprint_features(smiles) for smiles in train_df['smiles']])
                self.valid_mol_fps = np.array([get_fingerprint_features(smiles) for smiles in valid_df['smiles']])
                self.test_mol_fps = np.array([get_fingerprint_features(smiles) for smiles in test_df['smiles']])
                
                train_labels = train_df[self.task_columns].values                
                self.mlp_fp_model = MLPPredictor(in_features=self.train_mol_fps.shape[1], out_features=train_labels.shape[1], task_type=self.task_type)
                self.mlp_fp_model.fit(self.train_mol_fps, train_labels, verbose=verbose)
                
                for task_col in self.task_columns:
                    self.fingerprint_models[task_col] = self.mlp_fp_model                
            
        else:
            if model_type == 'RF':
                if self.task_type == 'classification':
                    from sklearn.ensemble import RandomForestClassifier
                    base_predictor = lambda: RandomForestClassifier(n_estimators=100, random_state=42, verbose=0)
                elif self.task_type == 'regression':
                    from sklearn.ensemble import RandomForestRegressor
                    base_predictor = lambda: RandomForestRegressor(n_estimators=100, random_state=42, verbose=0)
                else:
                    raise ValueError(f"Unsupported task type: {task_type}")
            elif model_type == 'GP':
                if self.task_type == 'classification':
                    from sklearn.gaussian_process import GaussianProcessClassifier
                    base_predictor = lambda: GaussianProcessClassifier(random_state=42)
                elif self.task_type == 'regression':
                    from sklearn.gaussian_process import GaussianProcessRegressor
                    base_predictor = lambda: GaussianProcessRegressor(random_state=42)
                else:
                    raise ValueError(f"Unsupported task type: {task_type}")
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            if verbose:
                print(f"Training {model_type} models for {len(self.task_columns)} tasks")
                task_iterator = tqdm(self.task_columns, desc="Training models")
            else:
                task_iterator = self.task_columns
                
            for task_col in task_iterator:
                train_task_mask = ~train_df[task_col].isna()
                train_task_labels = train_df.loc[train_task_mask, task_col].values
                
                if len(train_task_labels) == 0:
                    if verbose:
                        print(f"  Skipping task {task_col}: No valid training labels")
                    continue
                
                eval_features_mask = ~np.isnan(train_eval_features).any(axis=1)
                combined_mask = train_task_mask & pd.Series(eval_features_mask, index=train_df.index)
                
                if len(set(train_df.loc[combined_mask, task_col].values)) > 1:
                    if combined_mask.sum() > 0:
                        model = base_predictor()
                        model.fit(
                            train_eval_features[combined_mask.values], 
                            train_df.loc[combined_mask, task_col].values
                        )
                        self.eval_feature_models[task_col] = model
                else:
                    if verbose:
                        print(f"  Skipping task {task_col} for the representation: All labels are the same")
                
                # Train fingerprint model if needed
                if self.need_fingerprint_model and task_col not in self.fingerprint_models:
                    if not hasattr(self, 'train_mol_fps'):
                        self.train_mol_fps = np.array([get_fingerprint_features(smiles) for smiles in train_df['smiles']])
                        self.valid_mol_fps = np.array([get_fingerprint_features(smiles) for smiles in valid_df['smiles']])
                        self.test_mol_fps = np.array([get_fingerprint_features(smiles) for smiles in test_df['smiles']])
            
                    fp_model = base_predictor()
                    fp_model.fit(self.train_mol_fps[train_task_mask], train_task_labels)
                    self.fingerprint_models[task_col] = fp_model
                    
        # Print summary
        if verbose:
            print(f"\nTraining Summary ({model_type}-{self.task_type}):")
            total_tasks = len(self.task_columns)
            eval_count = len(self.eval_feature_models)
            fp_count = len(self.fingerprint_models)
            
            print(f"  Tasks: {total_tasks} total")
            print(f"  Evaluated feature models: {eval_count}/{total_tasks}")
            print(f"  Fingerprint models: {fp_count}/{total_tasks}")
            
            if model_type == 'MLP':
                print(f"  Note: Using single multitask MLP for all tasks")
            
        return self.eval_feature_models, self.fingerprint_models
    
    def evaluate(self, eval_feature_models=None, fingerprint_models=None, splits=None, verbose=False, return_predictions=True):
        """
        Evaluate models on validation and test datasets.
        
        Args:
            eval_feature_models (dict, optional): Dictionary of models trained on evaluated features.
                                                 If None, uses self.eval_feature_models.
            fingerprint_models (dict, optional): Dictionary of models trained on fingerprints.
                                                If None, uses self.fingerprint_models.
            splits (list, optional): List of splits to evaluate on. If None, uses ['valid', 'test'].
            verbose (bool): Whether to print detailed evaluation information.
            return_predictions (bool): Whether to return prediction dataframes.
            
        Returns:
            dict: Dictionary containing evaluation metrics for each split and task.
            dict (optional): Dictionary of prediction dataframes for each split.
        """        
        # Use class models if not provided
        eval_feature_models = eval_feature_models or self.eval_feature_models
        fingerprint_models = fingerprint_models or self.fingerprint_models
        splits = splits or ['valid', 'test']
        
        if self.task_merged_df is None:
            raise ValueError("Task data is not merged with feature data. Process data first with match_task_data.")
        
        if not eval_feature_models and not fingerprint_models:
            raise ValueError("No models available for evaluation. Train models first.")
        
        if isinstance(eval_feature_models, str):
            assert eval_feature_models == 'baseline'
            print('Evaluating on fingerprint model')
        
        # Initialize results dictionary
        results = {split: {} for split in splits}
        
        # Initialize predictions dictionary
        predictions = {split: None for split in splits}
        
        # Evaluate on each split
        for split in splits:
            if verbose:
                print(f"\nEvaluating on {split} set:")
            
            # Get data for current split
            split_df = self.task_merged_df[self.task_merged_df['split'] == split]
            
            if len(split_df) == 0:
                print(f"Warning: No data found for split '{split}'")
                continue
            
            # Create prediction dataframe with SMILES column and pre-allocate prediction columns
            pred_df = pd.DataFrame({'smiles': split_df['smiles']})
            
            # Pre-allocate all prediction columns at once
            pred_columns = {}
            if return_predictions:
                # Add ground truth columns
                for task_col in self.task_columns:
                    pred_columns[f'true_{task_col}'] = split_df[task_col]
                    pred_columns[f'pred_{task_col}'] = np.nan
            else:
                # Only prediction columns
                for task_col in self.task_columns:
                    pred_columns[f'pred_{task_col}'] = np.nan
                    
            # Add all columns at once
            pred_df = pd.concat([pred_df, pd.DataFrame(pred_columns, index=pred_df.index)], axis=1)
            
            eval_features = split_df[self.feature_cols].values if self.feature_cols else None
            if hasattr(self, f'{split}_mol_fps'):
                fingerprints = getattr(self, f'{split}_mol_fps')
            else:
                fingerprints = None
                if fingerprint_models and (eval_features is None or np.isnan(eval_features).any()):
                    if verbose:
                        print(f"Generating fingerprints for {split} set...")
                        fingerprints = np.array([
                            get_fingerprint_features(smiles) for smiles in 
                            tqdm(split_df['smiles'], desc=f"Generating {split} fingerprints")
                        ])
                    else:
                        fingerprints = np.array([get_fingerprint_features(smiles) for smiles in split_df['smiles']])
            
            # Evaluate each task
            task_iterator = tqdm(self.task_columns, desc=f"Evaluating tasks") if verbose else self.task_columns
            
            for task_idx, task_col in enumerate(task_iterator):
                task_mask = ~split_df[task_col].isna()
                if task_mask.sum() == 0:
                    if verbose:
                        print(f"  Skipping task {task_col}: No valid labels in {split} set")
                    # Add NaN column to prediction dataframe
                    pred_df[task_col] = np.nan
                    continue
                
                y_true = split_df.loc[task_mask, task_col].values                
                y_pred = np.full(len(split_df), np.nan)
                
                if eval_feature_models != 'baseline':
                    if task_col in eval_feature_models and eval_features is not None:
                        model = eval_feature_models[task_col]                    
                        eval_features_mask = ~np.isnan(eval_features).any(axis=1)
                        
                        if eval_features_mask.any():
                            if self.task_type == 'classification':
                                if isinstance(model, MLPPredictor):
                                    eval_predictions = model.predict_proba(eval_features[eval_features_mask])[:, task_idx]
                                else:
                                    outputs = model.predict_proba(eval_features[eval_features_mask])
                                    if outputs.shape[1] == 1:
                                        warn(f"Skip evaluating the representation for classification task {task_col}: Single output")
                                    else:
                                        eval_predictions = outputs[:, 1]
                            else:
                                if isinstance(model, MLPPredictor):
                                    eval_predictions = model.predict(eval_features[eval_features_mask])[:, task_idx]
                                else:
                                    eval_predictions = model.predict(eval_features[eval_features_mask])
                            y_pred[eval_features_mask] = eval_predictions

                if task_col in fingerprint_models and fingerprints is not None:
                    model = fingerprint_models[task_col]
                    nan_mask = np.isnan(y_pred)
                    if nan_mask.any():
                        if self.task_type == 'classification':
                            if isinstance(model, MLPPredictor):
                                fp_predictions = model.predict_proba(fingerprints[nan_mask])[:, task_idx]
                            else:
                                fp_predictions = model.predict_proba(fingerprints[nan_mask])[:, 1]
                        else:
                            if isinstance(model, MLPPredictor):
                                fp_predictions = model.predict(fingerprints[nan_mask])[:, task_idx]
                            else:
                                fp_predictions = model.predict(fingerprints[nan_mask])
                        # print('fp_predictions', fp_predictions.shape, 'y_pred', y_pred.shape, 'nan_mask', nan_mask.shape)
                        y_pred[nan_mask] = fp_predictions
                
                # Update prediction column (now pre-allocated)
                pred_df.loc[:, f"pred_{task_col}"] = y_pred
                
                # Compute metrics
                if np.isnan(y_pred).any():
                    print('task_col in fingerprint_models', task_col in fingerprint_models)
                    print('fingerprints is not None', fingerprints is not None)
                    raise ValueError(f"y_pred contains NaN for task {task_col}", 'y_pred', y_pred, 'y_true', split_df.loc[task_mask, task_col].values)
                
                metric_value = compute_metric(
                    y_true=split_df.loc[task_mask, task_col].values,
                    y_pred=y_pred[task_mask],
                    task_type=self.task_type
                )
                results[split][task_col] = metric_value
            
            # Store prediction dataframe
            predictions[split] = pred_df
            
            # After evaluating all tasks, calculate average
            if results[split]:
                # Filter out NaN values
                valid_metrics = [value for value in results[split].values() 
                                if not (isinstance(value, float) and np.isnan(value))]
                
                if valid_metrics:
                    # Calculate average of all task metrics
                    avg_metric = np.mean(valid_metrics)
                    results[split]['average'] = avg_metric
                    
                    if verbose:
                        metric_name = 'AUC' if self.task_type == 'classification' else 'MAE'
                        print(f"\n{split.capitalize()} set average {metric_name}: {avg_metric:.4f}")
                        print(f"Tasks evaluated: {len(valid_metrics)}/{len(self.task_columns)}")

        results_df = []
        for task in results['valid'].keys():
            if task != 'average':
                results_df.append({
                    'task': task,
                    'valid': results['valid'][task],
                    'test': results['test'][task]
                })
                
        results_df = pd.DataFrame(results_df)
        if eval_feature_models != 'baseline':
            self.evaluation_results = results_df
            self.prediction_dfs = predictions

        if return_predictions:
            return results_df, predictions
        else:
            return results_df

    def save_results(self, output_dir='output', results=None, model_type=None):
        if model_type is None:
            model_type = self.model_type
        if model_type is None:
            raise ValueError("Model type is not set. Either provide model_type or run train_predictor() first.")

        os.makedirs(output_dir, exist_ok=True)
        if results is None:
            results = self.evaluation_results
            
        # # Create list of task results
        # task_results = []
        # for task in results['valid'].keys():
        #     if task != 'average':
        #         task_results.append({
        #             'task': task,
        #             'valid': results['valid'][task],
        #             'test': results['test'][task]
        #         })
        # results_df = pd.DataFrame(task_results)
        
        output_results_path = os.path.join(output_dir, f"{self.task_name}_{model_type}_{self.task_type}_eval.csv")
        results.to_csv(output_results_path, index=False)

        print(f"Results saved to {output_results_path}")
        if self.prediction_dfs is not None:
            for split, pred_df in self.prediction_dfs.items():
                output_predictions_path = os.path.join(output_dir, f"{self.task_name}_{model_type}_{self.task_type}_{split}.csv")
                pred_df.to_csv(output_predictions_path, index=False)
                print(f"Predictions saved to {output_predictions_path}")
    
    def load_results(self, input_dir, task_name=None, model_type=None):
        if model_type is None:
            model_type = self.model_type
        if model_type is None:
            raise ValueError("Model type is not set. Either provide model_type or run train_predictor() first.")

        if task_name is None:
            task_name = self.task_name
            task_type = self.task_type
        elif task_name is not None:
            self.task_name = task_name
            task_type = self.task_configs[task_name]['task_type']
            self.task_type = task_type
        else:
            raise ValueError("Task name should be provided")
 
        input_path = os.path.join(input_dir, f"{task_name}_{model_type}_{task_type}_eval.csv")
        self.evaluation_results = pd.read_csv(input_path)
        print(f"Results loaded from {input_path}")
        self.task_name = task_name
        self.task_type = task_type
        
        self.prediction_dfs = {}
        input_path = os.path.join(input_dir, f"{task_name}_{model_type}_{task_type}_valid.csv")
        self.prediction_dfs['valid'] = pd.read_csv(input_path)
        print(f"Predictions loaded from {input_path}")
        
        input_path = os.path.join(input_dir, f"{task_name}_{model_type}_{task_type}_test.csv")
        self.prediction_dfs['test'] = pd.read_csv(input_path)
        print(f"Predictions loaded from {input_path}")
        return self.evaluation_results, self.prediction_dfs
    
    def plot_results(self, output_dir='output', results=None, model_type=None):
        if model_type is None:
            model_type = self.model_type
        if model_type is None:
            raise ValueError("Model type is not set. Either provide model_type or run train_predictor() first.")

        results = results or self.evaluation_results
        if results is None:
            raise ValueError("No evaluation results available. Run evaluate() first.")
        if isinstance(results, pd.DataFrame):
            results = {
                "valid": dict(zip(results["task"], results["valid"])),
                "test": dict(zip(results["task"], results["test"]))
            }

        os.makedirs(output_dir, exist_ok=True)
        if self.task_type is None:
            raise ValueError("Task type is not set. Run load_results() first.")
        metric_name = 'AUC' if self.task_type == 'classification' else 'MAE'
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Process each split
        for i, split in enumerate(['valid', 'test']):
            if split not in results:
                axes[i].text(0.5, 0.5, f"No data for {split} split", 
                            ha='center', va='center', transform=axes[i].transAxes)
                continue
            
            split_results = {k: v for k, v in results[split].items() if k != 'average'}
            df = pd.Series(split_results).dropna()
            
            if len(df) == 0:
                axes[i].text(0.5, 0.5, f"No valid data for {split} split", 
                            ha='center', va='center', transform=axes[i].transAxes)
                continue
            
            # Sort values (ascending for regression, descending for classification)
            df = df.sort_values(ascending=(self.task_type == 'regression'))
            
            # Plot bar chart
            ax = df.plot(kind='bar', ax=axes[i], color='skyblue')
            
            # Use task names from the task column if available
            if (isinstance(self.evaluation_results, pd.DataFrame) 
                and 'task' in self.evaluation_results.columns 
                and self.task_name not in ['toxcast']):
                task_mapping = dict(zip(self.evaluation_results.index, self.evaluation_results['task']))
                task_names = [task_mapping.get(idx, idx) for idx in df.index]
                ax.set_xticklabels(task_names, rotation=90, ha='center')
            else:
                task_names = df.index
                ax.set_xticklabels(task_names, rotation=90, ha='center')
            
            split_values = [v for k, v in results[split].items() 
                          if k != 'average' and not (isinstance(v, float) and np.isnan(v))]
            avg = np.mean(split_values) if split_values else None
            if avg is not None:
                axes[i].axhline(y=avg, color='r', linestyle='-', label=f'Avg: {avg:.3f}')
                axes[i].legend()
            
            # Set labels
            axes[i].set_title(f"{split.capitalize()} Set")
            axes[i].set_ylabel(metric_name)
            axes[i].set_xlabel("Tasks")
            
            # Adjust bottom margin to accommodate labels
            plt.subplots_adjust(bottom=0.3)
        
        # Add title and save
        plt.suptitle(f"{self.task_name} - {self.task_type.capitalize()}")
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{self.task_name}_{model_type}_{self.task_type}_results.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path

    # def plot_comparison_bar(self, baseline, target=None, baseline_name='baseline', target_name=None, output_dir='output'):
    #     """Plot comparison bar charts between baseline and target results.
        
    #     Args:
    #         baseline (pd.DataFrame): Baseline results DataFrame with columns ['task', 'valid', 'test']
    #         target (pd.DataFrame, optional): Target results DataFrame. If None, uses self.evaluation_results
    #         baseline_name (str): Name for the baseline model (default: 'baseline')
    #         target_name (str, optional): Name for the target model. If None, uses 'evaluated'
    #         output_dir (str): Directory to save the output plot (default: 'output')
            
    #     Returns:
    #         str: Path to the saved plot
    #     """
    #     if target is None:
    #         if self.evaluation_results is None:
    #             raise ValueError("No evaluation results available. Run evaluate() first or provide target results.")
    #         target = self.evaluation_results
        
    #     if target_name is None:
    #         target_name = 'evaluated'
        
    #     if not all(col in baseline.columns for col in ['task', 'valid', 'test']):
    #         raise ValueError("Baseline DataFrame must contain 'task', 'valid', and 'test' columns")
        
    #     # Ensure tasks match between baseline and target
    #     baseline_tasks = set(baseline['task'])
    #     target_tasks = set(target['task'])
    #     common_tasks = baseline_tasks.intersection(target_tasks)
        
    #     if len(common_tasks) == 0:
    #         raise ValueError("No common tasks found between baseline and target results")
        
    #     # Filter to common tasks and sort
    #     baseline = baseline[baseline['task'].isin(common_tasks)].sort_values('task')
    #     target = target[target['task'].isin(common_tasks)].sort_values('task')
        
    #     # Create figure with two subplots
    #     fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
    #     metric_name = 'AUC' if self.task_type == 'classification' else 'MAE'
    #     bar_width = 0.35
        
    #     # Process each split
    #     for i, split in enumerate(['valid', 'test']):
    #         tasks = baseline['task']
    #         x = np.arange(len(tasks))
            
    #         baseline_values = baseline[split]
    #         target_values = target[split]
            
    #         # Calculate differences
    #         differences = target_values - baseline_values
            
    #         # Plot bars
    #         axes[i].bar(x - bar_width/2, baseline_values, bar_width, label=baseline_name, color='lightblue')
    #         axes[i].bar(x + bar_width/2, target_values, bar_width, label=target_name, color='lightgreen')
            
    #         # Plot differences as red/green bars on top
    #         for j, diff in enumerate(differences):
    #             color = 'green' if diff > 0 else 'red'
    #             if diff > 0:
    #                 axes[i].arrow(x[j], baseline_values.iloc[j], 0, diff, 
    #                             color=color, alpha=0.5, width=0.03,
    #                             head_width=0.1, head_length=min(abs(diff)*0.2, abs(diff)))
    #             else:
    #                 axes[i].arrow(x[j], target_values.iloc[j], 0, -diff, 
    #                             color=color, alpha=0.5, width=0.03,
    #                             head_width=0.1, head_length=min(abs(diff)*0.2, abs(diff)))
            
    #         # Calculate and plot averages
    #         baseline_avg = baseline_values.mean()
    #         target_avg = target_values.mean()
    #         axes[i].axhline(y=baseline_avg, color='blue', linestyle='--', 
    #                        label=f'{baseline_name} Avg: {baseline_avg:.3f}')
    #         axes[i].axhline(y=target_avg, color='green', linestyle='--',
    #                        label=f'{target_name} Avg: {target_avg:.3f}')
            
    #         # Customize plot
    #         axes[i].set_title(f"{split.capitalize()} Set")
    #         axes[i].set_ylabel(metric_name)
    #         axes[i].set_xlabel("Tasks")
    #         axes[i].set_xticks(x)
    #         axes[i].set_xticklabels(tasks, rotation=90, ha='center')
    #         axes[i].legend()
            
    #         # Add difference annotations
    #         for j, diff in enumerate(differences):
    #             if abs(diff) > 0.01:  # Only show significant differences
    #                 axes[i].annotate(f'{diff:+.2f}', 
    #                                xy=(x[j], max(baseline_values.iloc[j], target_values.iloc[j])),
    #                                xytext=(0, 5), textcoords='offset points',
    #                                ha='center', va='bottom',
    #                                color='green' if diff > 0 else 'red',
    #                                fontsize=8)
        
    #     # Add overall title and adjust layout
    #     plt.suptitle(f"{self.task_name} - {self.task_type.capitalize()}: {baseline_name} vs {target_name}")
    #     plt.tight_layout()
        
    #     # Save plot
    #     os.makedirs(output_dir, exist_ok=True)
    #     output_path = os.path.join(output_dir, 
    #                               f"{self.task_name}_{baseline_name}_vs_{target_name}_comparison.png")
    #     plt.savefig(output_path, dpi=300, bbox_inches='tight')
    #     plt.close()
        
    #     return output_path
