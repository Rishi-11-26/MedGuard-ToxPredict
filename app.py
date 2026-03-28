import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from model_trainer import extract_molecular_features

# 1. Page Configuration
st.set_page_config(page_title="MedGuard AI | Toxicity Lab", layout="wide")

# 2. Header and Branding
st.title("🧪 MedGuard AI: Pharmacology Toxicity Predictor")
st.markdown("---")

# 3. Sidebar for Batch Uploads (Impressive for Scaling)
st.sidebar.header("📁 Batch Processing")
uploaded_file = st.sidebar.file_uploader("Upload CSV for High-Throughput Screening", type="csv")

# 4. Main Input Area
st.subheader("🔍 Molecular Analysis")
user_input = st.text_area(
    "Enter SMILES strings (one per line for multiple drugs):", 
    "CC(=O)OC1=CC=CC=C1C(=O)O\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    help="You can enter one or many chemical strings here."
)

if st.button("🚀 Run AI Analysis"):
    # Split input into a list of drugs
    smiles_list = [s.strip() for s in user_input.replace(',', '\n').split('\n') if s.strip()]
    
    if not smiles_list:
        st.warning("Please enter at least one SMILES string.")
    else:
        try:
            # Load the trained AI model
            model = joblib.load('tox_model.pkl')
            
            # Create columns based on number of drugs (max 3 per row for visibility)
            num_drugs = len(smiles_list)
            cols = st.columns(num_drugs if num_drugs <= 3 else 3)

            for i, smiles in enumerate(smiles_list):
                col_idx = i % 3
                with cols[col_idx]:
                    st.markdown(f"### Drug {i+1}")
                    mol = Chem.MolFromSmiles(smiles)
                    
                    if mol:
                        # Draw Molecule
                        img = Draw.MolToImage(mol, size=(300, 300))
                        st.image(img, caption=f"SMILES: {smiles[:15]}...", use_column_width=True)
                        
                        # Extract features and Predict
                        feat = extract_molecular_features(smiles)
                        if feat is not None:
                            prob = model.predict_proba(feat.reshape(1, -1))[0][1]
                            
                            # Display Result with Color Coding
                            if prob > 0.5:
                                st.error(f"⚠️ HIGH RISK: {prob*100:.1f}%")
                            else:
                                st.success(f"✅ LOW RISK: {prob*100:.1f}%")
                            
                            # Pharmacological Metrics
                            st.write(f"**LogP:** {Descriptors.MolLogP(mol):.2f}")
                            st.write(f"**Mass:** {Descriptors.MolWt(mol):.1f} Da")
                    else:
                        st.error(f"Invalid SMILES: {smiles}")
                
                # Create a new row of columns if needed
                if (i + 1) % 3 == 0 and (i + 1) < num_drugs:
                    st.markdown("---")
                    cols = st.columns(3)

        except FileNotFoundError:
            st.error("❌ Error: 'tox_model.pkl' not found. Please run 'model_trainer.py' first!")

# 5. Handling Batch CSV Uploads
if uploaded_file is not None:
    st.markdown("---")
    st.subheader("📊 Batch Results")
    batch_df = pd.read_csv(uploaded_file)
    if 'smiles' in batch_df.columns:
        # Show a progress bar for large files
        st.info("Screening large dataset... please wait.")
        # (This is where you would apply the model logic to the whole dataframe)
        st.dataframe(batch_df.head(10))
    else:
        st.error("CSV must contain a column named 'smiles'")