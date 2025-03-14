from molbench.benchmark import MolBench
from molbench.data_utils import merge_compound_and_features
import pandas as pd
import os

# Set up cache directory
cache_dir = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(cache_dir, exist_ok=True)
profiles_path = "data/profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony.parquet"
compound_path = "data/compound.csv.gz"
cache_file = os.path.join(cache_dir, "merged_data.parquet")
if os.path.exists(cache_file):
    merged_df = pd.read_parquet(cache_file)
    print(f"Loaded cached data with shape: {merged_df.shape}")
else:
    profiles_df = pd.read_parquet(profiles_path)
    print(f"Loaded profiles data with shape: {profiles_df.shape}")
    compound_df = pd.read_csv(compound_path, compression='gzip')
    print(f"Loaded compound data with shape: {compound_df.shape}")
    
    # Use the merge_compound_and_features function to process and merge the dataframes
    merged_df = merge_compound_and_features(
        compound_df=compound_df,
        feature_df=profiles_df,
        merge_key='Metadata_JCP2022',
        inchikey_col='Metadata_InChIKey',
        smiles_col='Metadata_SMILES',
        feature_prefix='X_',
        output_inchikey_col='inchikey',
        output_smiles_col='smiles',
        reduction='mean',
        keep_other_cols=False
    )
    
    # Save to cache
    print(f"Saving merged data to cache: {cache_file}")
    merged_df.to_parquet(cache_file, index=False)

# Display information about the merged dataframe
print(f"Final merged dataframe shape: {merged_df.shape}")
print(merged_df.head())

# List of all tasks and models to test
tasks = ["chembl2k", "broad6k", "toxcast", "biogen3k"]
models = ["RF", "MLP", "GP"]

# Create output directory if it doesn't exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Prepare summary list
summary_results = []

# Loop through all tasks
for task in tasks:
    print(f"\n{'='*50}")
    print(f"Processing task: {task}")
    print(f"{'='*50}")
        
    # Load task data
    bench = MolBench()
    bench.load_task_data(task)
    print(f'Loaded task data: {bench.task_df.shape}')
    
    # Match task data with features
    matched_df = bench.match_task_data(merged_df)
    print(f'Matched task data: {matched_df.shape}')
    
    # Loop through all models
    for model_type in models:
        print(f"\n{'-'*30}")
        print(f"Training with model: {model_type}")
        print(f"{'-'*30}")
        
        # Train models
        feature_models, fingerprint_models = bench.train_predictor(model_type=model_type, verbose=False)
        
        # Evaluate models
        results, predictions = bench.evaluate(feature_models, fingerprint_models, verbose=False, return_predictions=True)
        
        # Compute averages
        valid_avg = results['valid'].mean()
        test_avg = results['test'].mean()
        print(f"Results for {task} using {model_type}: valid avg: {valid_avg}, test avg: {test_avg}")
        
        # Append results to summary list
        summary_results.append({
            "task": task,
            "model": model_type,
            "valid_avg": valid_avg,
            "test_avg": test_avg
        })
        
        # Save results
        bench.save_results(output_dir, results, model_type=model_type)
        bench.load_results(output_dir, task_name=task, model_type=model_type)
        bench.plot_results(output_dir)

# Convert summary results to DataFrame and save as CSV
summary_df = pd.DataFrame(summary_results)
summary_csv_path = os.path.join(output_dir, "summary.csv")
summary_df.to_csv(summary_csv_path, index=False)
print(f"Summary results saved to {summary_csv_path}")


# from molbench.benchmark import MolBench
# from molbench.data_utils import merge_compound_and_features
# import pandas as pd
# import os

# # Set up cache directory
# cache_dir = os.path.join(os.path.dirname(__file__), "cache")
# os.makedirs(cache_dir, exist_ok=True)
# profiles_path = "data/profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony.parquet"
# compound_path = "data/compound.csv.gz"
# cache_file = os.path.join(cache_dir, f"merged_data.parquet")
# if os.path.exists(cache_file):
#     merged_df = pd.read_parquet(cache_file)
#     print(f"Loaded cached data with shape: {merged_df.shape}")
# else:
    
#     profiles_df = pd.read_parquet(profiles_path)
#     print(f"Loaded profiles data with shape: {profiles_df.shape}")
#     compound_df = pd.read_csv(compound_path, compression='gzip')
#     print(f"Loaded compound data with shape: {compound_df.shape}")

#     # Use the merge_compound_and_features function to process and merge the dataframes
#     merged_df = merge_compound_and_features(
#         compound_df=compound_df,
#         feature_df=profiles_df,
#         merge_key='Metadata_JCP2022',
#         inchikey_col='Metadata_InChIKey',
#         smiles_col='Metadata_SMILES',
#         feature_prefix='X_',
#         output_inchikey_col='inchikey',
#         output_smiles_col='smiles',
#         reduction='mean',
#         keep_other_cols=False
#     )
    
#     # Save to cache
#     print(f"Saving merged data to cache: {cache_file}")
#     merged_df.to_parquet(cache_file, index=False)

# # Display information about the merged dataframe
# print(f"Final merged dataframe shape: {merged_df.shape}")
# print(merged_df.head())

# # List of all tasks and models to test
# tasks = ["chembl2k", "broad6k", "toxcast", "biogen3k"]
# models = ["RF", "MLP", "GP"]
# # tasks = ["biogen3k"]
# # models = ["MLP"]

# # Create output directory if it doesn't exist
# os.makedirs("output", exist_ok=True)

# # Loop through all tasks
# for task in tasks:
#     print(f"\n{'='*50}")
#     print(f"Processing task: {task}")
#     print(f"{'='*50}")
        
#     # Load task data
#     bench = MolBench()
#     bench.load_task_data(task)
#     print(f'Loaded task data: {bench.task_df.shape}')
    
#     # Match task data with features
#     matched_df = bench.match_task_data(merged_df)
#     print(f'Matched task data: {matched_df.shape}')
    
#     # Loop through all models
#     for model_type in models:
#         print(f"\n{'-'*30}")
#         print(f"Training with model: {model_type}")
#         print(f"{'-'*30}")
        
#         # Train models
#         feature_models, fingerprint_models = bench.train_predictor(model_type=model_type, verbose=False)

#         # baseline_dict = bench.evaluate('baseline', fingerprint_models, verbose=False, return_predictions=False)
#         # baseline_df = pd.DataFrame(baseline_dict)
#         # Evaluate models
#         results, predictions = bench.evaluate(feature_models, fingerprint_models, verbose=False, return_predictions=True)
#         print(f"Results for {task} using {model_type}:")
#         print('task: ', task, 'model: ', model_type, 'valid avg: ', results['valid'].mean(), 'test avg: ', results['test'].mean())
        
#         # Save results
#         bench.save_results(f'output', results, model_type=model_type)
#         bench.load_results(f'output', task_name=task, model_type=model_type)
#         bench.plot_results(f'output')
#         # bench.plot_comparison_bar(
#         #     baseline=baseline_df,
#         #     target=results,
#         #     baseline_name='Morgan',
#         #     target_name='CellMorph',
#         #     output_dir='output'
#         # )