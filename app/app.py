import streamlit as st
import numpy as np
from src.pipeline import load_pipeline
from src.utils import draw_molecule_2d, draw_molecule_3d
from src.data_prep import collect_smiles

st.set_page_config(page_title="ChemTaste Agent", page_icon="🧪", layout="centered")

pipe = load_pipeline("models/taste_agent_pipeline.joblib")

st.title("🧪 ChemTaste Agent")
st.write("Enter a molecule (SMILES or name), and the agent will predict its taste!")

mode = st.radio("Input type:", ["Molecule name", "SMILES string"])
name = st.text_input("Molecule:")

if st.button("Predict") and name.strip():
    try:
        smiles = name.strip()  # TODO: convert molecule name from PubChem to SMILES
        st.info(f"Using SMILES: `{smiles}`")

        # Visualization
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("2D structure")
            st.image(draw_molecule_2d(smiles))
        with col2:
            st.subheader("3D structure")
            st.components.v1.html(draw_molecule_3d(smiles)._make_html(), height=400)

        # Prediction
        y_proba = pipe.predict_proba([smiles])[0]
        classes = pipe.named_steps["clf"].classes_
        pred = classes[np.argmax(y_proba)]
        st.success(f"Predicted taste: **{pred}**")

    except Exception as e:
        st.error(str(e))
