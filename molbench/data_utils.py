import pandas as pd
from rdkit import Chem

def generate_inchikey(smiles):
    """
    Generate InChIKey from SMILES string and return the first 14 characters (molecular part).
    
    Args:
        smiles (str): SMILES representation of a molecule
        
    Returns:
        str: First 14 characters of InChIKey or None if conversion fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            inchikey = Chem.MolToInchiKey(mol)
            return inchikey[:14]  # Return first 14 characters (molecular part)
        return None
    except:
        return None

def ensure_inchikey_column(dataframe, smiles_column='smiles'):
    """
    Check if dataframe has an 'inchikey' column, and if not, create it from SMILES.
    
    Args:
        dataframe (pd.DataFrame): DataFrame to check/modify
        smiles_column (str): Name of the column containing SMILES strings
        
    Returns:
        pd.DataFrame: DataFrame with inchikey column added if it didn't exist
    """
    if 'inchikey' not in dataframe.columns:
        if smiles_column not in dataframe.columns:
            raise ValueError(f"DataFrame does not contain a '{smiles_column}' column to generate InChIKeys")
        
        print(f"Generating InChIKey column from {smiles_column}...")
        dataframe['inchikey'] = dataframe[smiles_column].apply(generate_inchikey)
        
    return dataframe

def merge_compound_and_features(
    compound_df, 
    feature_df, 
    merge_key='Metadata_JCP2022',
    inchikey_col='Metadata_InChIKey',
    smiles_col='Metadata_SMILES',
    feature_prefix='X_',
    output_inchikey_col='inchikey',
    output_smiles_col='smiles',
    reduction='mean',
    keep_other_cols=False
):
    """
    Process and merge compound data with feature data.
    
    Args:
        compound_df (pd.DataFrame): DataFrame containing compound information
        feature_df (pd.DataFrame): DataFrame containing feature data
        merge_key (str): Column name to use for merging the dataframes
        inchikey_col (str): Column name in compound_df containing InChIKeys, or None to compute from SMILES
        smiles_col (str): Column name in compound_df containing SMILES strings
        feature_prefix (str): Prefix used to identify feature columns
        output_inchikey_col (str): Name for the output InChIKey column
        output_smiles_col (str): Name for the output SMILES column
        reduction (str): Method to use for aggregating duplicate inchikeys ('mean', 'median', 'max', 'min')
        keep_other_cols (bool): Whether to keep the other columns in the compound_df
        
    Returns:
        pd.DataFrame: Merged dataframe with specified column order, containing only rows where
                     all features and SMILES are not null, with duplicate inchikeys aggregated
    """
    # Validate inputs
    if merge_key not in compound_df.columns or merge_key not in feature_df.columns:
        raise ValueError(f"Merge key '{merge_key}' not found in both dataframes")
    
    if smiles_col not in compound_df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in compound dataframe")
    
    valid_reductions = ['mean', 'median', 'max', 'min']
    if reduction not in valid_reductions:
        raise ValueError(f"Invalid reduction method: {reduction}. Must be one of {valid_reductions}")
    
    # Process compound data to extract required columns
    processed_compound_df = compound_df.copy()
    processed_compound_df[output_smiles_col] = processed_compound_df[smiles_col]
    
    # Handle InChIKey - either extract from existing column or compute from SMILES
    if inchikey_col is not None and inchikey_col in compound_df.columns:
        processed_compound_df[output_inchikey_col] = processed_compound_df[inchikey_col].str[:14]
    else:
        processed_compound_df[output_inchikey_col] = processed_compound_df[output_smiles_col].apply(
            generate_inchikey
        )
    
    # Keep only necessary columns for merging
    processed_compound_df = processed_compound_df[[merge_key, output_smiles_col, output_inchikey_col]]
    
    # Merge the dataframes
    merged_df = pd.merge(
        feature_df, 
        processed_compound_df, 
        on=merge_key, 
        how='left'
    )
    
    # Reorder columns to have merge_key, smiles, inchikey first, followed by feature columns
    feature_cols = [col for col in merged_df.columns if col.startswith(feature_prefix)]
    other_cols = [col for col in merged_df.columns 
                 if not col.startswith(feature_prefix) 
                 and col not in [merge_key, output_smiles_col, output_inchikey_col]]
    
    if keep_other_cols:
        ordered_cols = [merge_key, output_smiles_col, output_inchikey_col] + other_cols + feature_cols
    else:
        ordered_cols = [merge_key, output_smiles_col, output_inchikey_col] + feature_cols
    
    merged_df = merged_df[ordered_cols]
    valid_rows = ~merged_df[output_smiles_col].isna()
    
    # Then check that all feature columns are not null
    for col in feature_cols:
        valid_rows = valid_rows & ~merged_df[col].isna()
    
    # Apply the filter
    filtered_df = merged_df[valid_rows].copy()
    
    # Check for duplicate inchikeys
    duplicate_inchikeys = filtered_df[filtered_df.duplicated(subset=[output_inchikey_col], keep=False)]
    if len(duplicate_inchikeys) > 0:
        # Identify columns to aggregate (feature columns) and columns to keep as is
        agg_dict = {}
        
        # For feature columns, use the specified reduction method
        for col in feature_cols:
            agg_dict[col] = reduction
        
        # For other columns, keep the first value
        if keep_other_cols: 
            print(f"Keeping the first for columns: {other_cols}")
            for col in [merge_key, output_smiles_col] + other_cols:
                agg_dict[col] = 'first'
        else:
            for col in [merge_key, output_smiles_col]:
                agg_dict[col] = 'first'
        
        # Group by inchikey and aggregate
        aggregated_df = filtered_df.groupby(output_inchikey_col).agg(agg_dict).reset_index()
        
        # Reorder columns to match original order
        aggregated_df = aggregated_df[ordered_cols]
        
        return aggregated_df
    else:
        return filtered_df