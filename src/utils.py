from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import py3Dmol

def draw_molecule_2d(smiles, size=(400,400)):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Draw.MolToImage(mol, size=size)

def draw_molecule_3d(smiles, size=(400,400)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    pdb = Chem.MolToPDBBlock(mol)
    viewer = py3Dmol.view(width=size[0], height=size[1])
    viewer.addModel(pdb, "pdb")
    viewer.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
    viewer.zoomTo()
    return viewer
