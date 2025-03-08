import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

def get_fingerprint_features(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    
    morgan_gen = GetMorganGenerator(radius=radius, fpSize=nBits)
    fingerprint = morgan_gen.GetFingerprint(mol)
    
    array = np.zeros((nBits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    
    return array