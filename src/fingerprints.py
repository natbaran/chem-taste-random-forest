import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

def smiles_to_fingerprint(smiles: str, radius=2, nBits=1024):
    """Converts a SMILES string to a Morgan fingerprint as a numpy array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"❌ Invalid SMILES: {smiles}")
    
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    fp = gen.GetFingerprint(mol)
    
    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    
    return arr

def smiles_list_to_fingerprints(X):
    return np.vstack([smiles_to_fingerprint(s) for s in X])

