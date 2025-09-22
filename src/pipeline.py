import joblib
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from .fingerprints import smiles_list_to_fingerprints  # importujemy gotową top-level funkcję

def build_pipeline():
    """
    Builds a scikit-learn pipeline:
    1. Converts SMILES strings to fingerprints.
    2. Trains a RandomForestClassifier.
    """
    feat = FunctionTransformer(smiles_list_to_fingerprints)  # zamiast lambdy
    pipe = Pipeline([
        ("feat", feat),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42))
    ])
    return pipe

def save_pipeline(pipe, path="models/taste_agent_pipeline.joblib"):
    """Saves the pipeline to a file using joblib."""
    joblib.dump(pipe, path)
    print(f"✅ Pipeline saved: {path}")

def load_pipeline(path="models/taste_agent_pipeline.joblib"):
    """Loads a pipeline from a joblib file."""
    return joblib.load(path)
