from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from copy import deepcopy
import numpy as np
from collections import defaultdict
from joblib import Parallel, delayed

def get_scaffold(mol):
    """Extracts the Murcko Scaffold from a molecule."""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def parallel_scaffold_computation(molecule, molecule_id):
    """Computes scaffold for a single molecule in parallel."""
    scaffold = get_scaffold(molecule)
    return scaffold, molecule, molecule_id

def cluster_molecules_by_scaffold(
    molecules, all_data_id, n_jobs=-1, remove_single=True, flatten_id=True
):
    """Clusters molecules based on their scaffolds using parallel processing."""
    # Ensure molecules and IDs are paired correctly
    paired_results = Parallel(n_jobs=n_jobs)(
        delayed(parallel_scaffold_computation)(mol, molecule_id)
        for mol, molecule_id in zip(molecules, all_data_id)
    )

    # Initialize dictionaries for batches and IDs
    batch = defaultdict(list)
    batched_data_id = defaultdict(list)

    # Process results to fill batch and batched_data_id dictionaries
    for scaffold, mol, molecule_id in paired_results:
        batch[scaffold].append(mol)
        batched_data_id[scaffold].append(molecule_id)

    # Optionally remove clusters with only one molecule
    if remove_single:
        batch = {scaffold: mols for scaffold, mols in batch.items() if len(mols) > 1}
        batched_data_id = {
            scaffold: ids for scaffold, ids in batched_data_id.items() if len(ids) > 1
        }

    # Convert dictionaries to lists for output
    scaffolds = list(batch.keys())
    batch = list(batch.values())
    batched_data_id = list(batched_data_id.values())
    if flatten_id:
        batched_data_id = [idd for batch in batched_data_id for idd in batch]
        batched_data_id = np.array(batched_data_id)

    return scaffolds, batch, batched_data_id

def append_scaffold_split_column(data_df, train_ratio=0.6, valid_ratio=0.15, test_ratio=0.25, random_seed=3, verbose=True):
    """
    Splits a dataframe of molecules into scaffold-based clusters and adds a 'split' column.
    
    Args:
        data_df (pd.DataFrame): DataFrame containing molecule data with a 'smiles' column
        train_ratio (float): Proportion of data for training set
        valid_ratio (float): Proportion of data for validation set
        test_ratio (float): Proportion of data for test set
        random_seed (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Original dataframe with an additional 'split' column indicating 
                     'train', 'valid', or 'test' for each row
    """
    # Validate input
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + valid_ratio + test_ratio}")
    
    if "smiles" not in data_df.columns:
        raise ValueError("DataFrame must contain a 'smiles' column")
    
    # Create a copy of the dataframe to avoid modifying the original
    result_df = data_df.copy()
    
    # Get smiles from the dataframe
    smiles_list = result_df["smiles"]
    indices = list(range(len(smiles_list)))
    
    # Convert SMILES to RDKit molecules
    mol_list = []
    valid_indices = []
    
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol_list.append(mol)
            valid_indices.append(i)
    
    # Get scaffold clusters
    scaffold_names, _, batched_id = cluster_molecules_by_scaffold(mol_list, valid_indices, remove_single=False, flatten_id=False)
    
    # Calculate cutoffs
    train_cutoff = int(train_ratio * len(result_df))
    valid_cutoff = int(valid_ratio * len(result_df)) + train_cutoff
    
    # Initialize split column with None
    result_df['split'] = None
    
    # Shuffle the clusters
    inds_all = deepcopy(batched_id)
    np.random.seed(random_seed)
    np.random.shuffle(inds_all)
    
    # Assign splits
    idx_count = 0
    for inds_list in inds_all:
        for ind in inds_list:
            if idx_count < train_cutoff:
                result_df.loc[ind, 'split'] = 'train'
            elif idx_count < valid_cutoff:
                result_df.loc[ind, 'split'] = 'valid'
            else:
                result_df.loc[ind, 'split'] = 'test'
            idx_count += 1
    
    # Check if any rows didn't get assigned (this shouldn't happen)
    unassigned = result_df[result_df['split'].isna()]
    if len(unassigned) > 0:
        print(f"Warning: {len(unassigned)} rows were not assigned a split")
    
    # Print split statistics
    train_count = (result_df['split'] == 'train').sum()
    valid_count = (result_df['split'] == 'valid').sum()
    test_count = (result_df['split'] == 'test').sum()
    
    if verbose:
        print(f"Split statistics:")
        print(f"  Train: {train_count} ({train_count/len(result_df):.2%})")
        print(f"  Valid: {valid_count} ({valid_count/len(result_df):.2%})")
        print(f"  Test:  {test_count} ({test_count/len(result_df):.2%})")
    
    return result_df
