import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def extract_molecular_features(smiles):
    """Converts SMILES into a mathematical vector for the AI."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # structural 'DNA' (Morgan Fingerprint)
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        
        # Physical properties required by Track A (LogP, Weight, QED)
        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        qed = Descriptors.qed(mol) 
        
        return np.append(fp, [logp, mw, qed])
    except Exception:
        return None

def train_model():
    print("📂 Loading Tox21 Dataset...")
    try:
        # Ensure your downloaded file is named 'tox21.csv'
        df = pd.read_csv('tox21.csv')
    except Exception as e:
        print(f"❌ File Error: {e}. Make sure tox21.csv is in the folder.")
        return

    # --- DATA CLEANING (Fixes ValueError: Invalid classes) ---
    # 1. Drop rows missing critical data
    df = df.dropna(subset=['smiles', 'NR-AR'])
    
    # 2. Force labels to be exactly 0 or 1 
    df['NR-AR'] = pd.to_numeric(df['NR-AR'], errors='coerce')
    df = df.dropna(subset=['NR-AR'])
    df['NR-AR'] = df['NR-AR'].astype(int)
    
    # Filter for only 0 and 1 (removes any other noise)
    df = df[df['NR-AR'].isin([0, 1])]

    print(f"🧪 Processing {len(df.head(2000))} molecules...")
    features = []
    labels = []
    
    for idx, row in df.head(2000).iterrows():
        feat = extract_molecular_features(row['smiles'])
        if feat is not None:
            features.append(feat)
            labels.append(row['NR-AR'])
            
    X = np.array(features)
    y = np.array(labels)
    
    if len(np.unique(y)) < 2:
        print("❌ Error: Not enough variety in data labels to train.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("🤖 Training XGBoost Model...")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'tox_model.pkl')
    print("✅ Success! 'tox_model.pkl' created.")

if __name__ == "__main__":
    train_model()