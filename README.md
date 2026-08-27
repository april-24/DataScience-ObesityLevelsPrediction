# Obesity Level Classification — BMDS2003

This deployment folder contains:

- `obesity_classification_project.ipynb` — complete CRISP-DM notebook.
- `app.py` — Streamlit deployment prototype.
- `obesity.csv` — supplied dataset.
- `requirements.txt` — Streamlit Cloud dependencies.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `obesity.csv`, and `requirements.txt`.
3. Deploy the repository on Streamlit Community Cloud.
4. Set the main file to `app.py`.

The app contains four sections: Overview, Exploratory Data Analysis, Model Comparison, and Single-Record Prediction.

## Assignment alignment

The notebook covers:
- Business Understanding
- Data Understanding
- Data Preparation
- Four classification models
- Baseline model (Decision Tree)
- Parameter tuning using stratified cross-validation
- Accuracy, precision, recall, and weighted F1 evaluation
- Confusion matrices
- Feature importance for tree models
- Limitations, conclusion, and references
