# 🧪 MedGuard AI: Pharmacology Toxicity Predictor
**A Computational Toxicology Tool for SPIRIT'26 CodeCure Hackathon**

MedGuard AI is an *in silico* screening tool designed to predict the toxicity of chemical compounds. By leveraging Machine Learning and Chemoinformatics, it identifies hazardous structural markers in potential drug candidates before they reach the clinical trial stage.

## 🚀 Features
- **Real-time Molecular Visualization:** Uses RDKit to render 2D chemical structures from SMILES strings.
- **AI-Powered Risk Assessment:** Employs an XGBoost Classifier trained on the **Tox21 Dataset**.
- **Pharmacological Metrics:** Calculates Molecular Weight, LogP (Lipophilicity), and QED (Drug-likeness).
- **Batch Processing:** Supports high-throughput screening of multiple compounds simultaneously.

## 🛠️ Tech Stack
- **Language:** Python 3.13
- **Chemoinformatics:** RDKit
- **Machine Learning:** XGBoost, Scikit-learn
- **Dashboard:** Streamlit
- **Data Source:** Tox21 Challenge Dataset (Kaggle)

## 📈 Technical Workflow
1. **Data Engineering:** Extracted 2048-bit **Morgan Fingerprints** to represent molecular "DNA."
2. **Model Training:** Optimized an XGBoost model to handle the high-dimensional sparse data of chemical structures.
3. **Feature Importance:** Identified that Lipophilicity (LogP) and specific functional groups are the primary drivers of toxicity in the dataset.

