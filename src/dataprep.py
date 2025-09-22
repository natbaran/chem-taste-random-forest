import os
import json
import requests
import pandas as pd
from rdkit import Chem

def collect_smiles(df, cache_path="smiles_cache.json"):
    """Fetches CID + IUPAC name for SMILES and stores them in a JSON cache."""
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    for s in df['Canonicalized SMILES']:
        if s in cache:
            continue
        try:
            # --- PubChem: CID
            url_cid = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{s}/cids/JSON'
            r = requests.get(url_cid)
            if r.status_code != 200:
                continue
            cids = r.json().get('IdentifierList', {}).get('CID', [])
            if not cids:
                continue
            cid = cids[0]

            # --- PubChem: IUPAC name
            url_name = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName/JSON'
            r = requests.get(url_name)
            if r.status_code != 200:
                continue
            name = r.json().get('PropertyTable', {}).get('Properties', [{}])[0].get('IUPACName')

            if name:
                cache[s] = name
                with open(cache_path, "w") as f:
                    json.dump(cache, f, indent=2)
        except:
            continue
    return cache


def clean_dataset(df, cache):
    """Cleans the dataset: removes invalid SMILES and filters for sweet/sour/bitter."""
    df['IUPAC_name'] = df['Canonicalized SMILES'].map(cache)
    df = df[df['IUPAC_name'].notna()].drop_duplicates(subset=('IUPAC_name','Canonicalized SMILES'))

    problematic_smiles = []
    for s in df['Canonicalized SMILES']:
        mol = Chem.MolFromSmiles(s)
        if mol:
            for atom in mol.GetAtoms():
                # Detect isolated hydrogen atoms (H with degree 0)
                if atom.GetAtomicNum() == 1 and atom.GetDegree() == 0:
                    problematic_smiles.append(s)
                    break

    df_clean = df[~df['Canonicalized SMILES'].isin(problematic_smiles)]
    df_filtered = df_clean[df_clean['Canonicalized Taste'].isin(['sweet','sour','bitter'])]

    # Balance class distribution
    min_class_size = df_filtered[df_filtered['Canonicalized Taste']!='sweet']\
                        .groupby('Canonicalized Taste').size().min()
    df_sweet_sampled = df_filtered[df_filtered['Canonicalized Taste']=='sweet']\
                        .sample(n=min_class_size, random_state=42)
    df_balanced = pd.concat([df_sweet_sampled,
                             df_filtered[df_filtered['Canonicalized Taste'].isin(['sour','bitter'])]])\
                             .reset_index(drop=True)
    return df_balanced
