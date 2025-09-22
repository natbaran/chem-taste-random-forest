# 🧪 ChemTaste Agent
Interactive agent to predict the taste of chemical molecules and visualize their 2D and 3D structures using RDKit and py3Dmol. 

## Features
- Interactive GUI using Streamlit
- Predict taste of molecules from SMILES or IUPAC names
- Visualize 2D molecular structures
- Visualize 3D molecular structures (Ball & Stick)
- Works for sweet, sour, and bitter tastes
- Display probability distribution for each taste class

## How to run
1. Clone this repository:
https://github.com/natbaran/chem-taste-random-forest

2. Install required packages:
pip install -r requirements.txt

3. Launch the Streamlit app:
streamlit run app/app.py

4. Open the URL provided by Streamlit in your browser (http://localhost:8501).

## Project structure
.
├── app/ # Streamlit app code
├── data/ # Cleaned and preprocessed dataset
├── models/ # Saved Random Forest model and pipeline (.joblib)
├── src/ # Helper scripts and visualization code
│ ├── dataprep.py
│ ├── fingerprints.py
│ ├── pipeline.py
│ ├── roc_utils.py
│ ├── run_pca_plot.ipynb
│ ├── run_roc_utils.ipynb
│ ├── train_rf.py
│ ├── training.py
│ └── utils.py
├── requirements.txt
└── README.md

## Technologies
- Python 3.10+
- Streamlit – interactive GUI
- pandas, numpy – data processing
- scikit-learn – Random Forest, PCA
- matplotlib – plots
- RDKit – molecular fingerprints
- PubChemPy – PubChem integration
- py3Dmol – 3D molecule visualization
- joblib – save/load pipelines

## Author
Natalia Baran ✨  
Contact: https://www.linkedin.com/in/natalia-baran-467277208/


