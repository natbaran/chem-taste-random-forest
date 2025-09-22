import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

# -----------------------------
# Top-level function to convert SMILES -> fingerprint
# -----------------------------
def smiles_to_fingerprint(smiles: str, radius=2, nBits=1024):
    """Converts SMILES -> Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"❌ Invalid SMILES: {smiles}")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# -----------------------------
# Named function for FunctionTransformer
# -----------------------------
def smiles_list_to_fingerprints(X):
    """Convert a list of SMILES to a 2D numpy array of fingerprints."""
    return np.vstack([smiles_to_fingerprint(s) for s in X])

# -----------------------------
# Pipeline: convert SMILES to fingerprints, then train RandomForest
# -----------------------------
feat = FunctionTransformer(smiles_list_to_fingerprints)
pipe = Pipeline([
    ("feat", feat),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42))
])

# -----------------------------
# Load preprocessed and balanced dataset
# -----------------------------
df_balanced = pd.read_csv("df_to_RF.csv")  # CSV with cleaned and balanced data
X_train_smiles = df_balanced["Canonicalized SMILES"].tolist()
y_train = df_balanced["Canonicalized Taste"].values

# -----------------------------
# Train pipeline and save model
# -----------------------------
pipe.fit(X_train_smiles, y_train)
joblib.dump(pipe, "taste_agent_pipeline.joblib")
print("RandomForest model saved as 'taste_agent_pipeline.joblib'")
