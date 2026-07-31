# Streamlit Deployment Prototype — Obesity Level Predictor

This folder is a **self-contained, deployable Streamlit app**. Everything it needs (trained
models, label encoder, metadata, cleaned dataset) is already inside `models/`.

## Folder contents

```
streamlit_app/
├── app.py                  <- main Streamlit application (entry point)
├── requirements.txt        <- pinned Python dependencies
├── .streamlit/
│   └── config.toml         <- theme settings
└── models/
    ├── decision_tree_pipeline.pkl   <- baseline model (Kyra)
    ├── random_forest_pipeline.pkl   <- Model 2 (Liping)
    ├── svm_pipeline.pkl             <- Model 3 (Wenhsuan)
    ├── knn_pipeline.pkl             <- Model 4 (Gladys)
    ├── label_encoder.pkl            <- encodes/decodes the 7 obesity classes
    ├── feature_metadata.json        <- feature ranges/categories used to build the input form
    ├── model_comparison.csv         <- pre-computed metrics table (from the notebook)
    └── obesity_cleaned.csv          <- cleaned dataset used by the Data Exploration tab
```

The app has **four distinct tabs** (not mixed together):

| Tab | What it shows |
|---|---|
| 🏠 About | Business problem, dataset summary, CRISP-DM workflow, team/model ownership |
| 🔮 Prediction | A form to enter a person's attributes, choose a model, and get a live prediction + probability chart + recommendation |
| 📊 Data Exploration | Filters, histograms, pie chart, scatterplot, boxplots, correlation heatmap, and an interactive OLAP-style pivot-table builder |
| 📈 Model Performance | Comparison table, inner (train vs test) and outer (model vs model) comparison charts, live confusion matrix + ROC curves per model |

## 1. Run it locally first (recommended sanity check)

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

It should open automatically at `http://localhost:8501`.

## 2. Deploy to Streamlit Community Cloud via GitHub

**Step 1 — Push this folder to GitHub**

The easiest option is to make the **contents of `streamlit_app/`** the root of your GitHub repo
(so `app.py` sits at the repo root), for example:

```bash
cd streamlit_app
git init
git add .
git commit -m "Obesity level predictor - Streamlit prototype"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

> If you'd rather push the **whole** project (notebook + app together) to one repo, that's fine
> too — just remember to set the **Main file path** in Streamlit Cloud to
> `streamlit_app/app.py` (see Step 3).

**Step 2 — Go to Streamlit Community Cloud**

1. Visit [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **"New app"**.
3. Select your repository and branch.

**Step 3 — Set the main file path**

- If `app.py` is at the repo root: enter `app.py`.
- If you pushed the whole project: enter `streamlit_app/app.py`.

**Step 4 — Deploy**

Click **Deploy**. The first build installs everything in `requirements.txt` and can take a
few minutes (the Random Forest model file is ~9 MB, which is fine for GitHub/Streamlit Cloud,
but the initial `git push` may take a little longer than usual).

**Step 5 — Verify**

Once deployed, click through all four tabs, submit a prediction, and check the Model
Performance tab loads its charts. If something fails, check the "Manage app" logs panel in
Streamlit Cloud — it will point to the exact line/exception.

## Notes on reproducibility

- The **Model Performance** tab does not just show a static table — it also **recomputes** a
  live confusion matrix and ROC curves by rebuilding the exact same train/test split used in
  the notebook (same `random_state=42`, `test_size=0.20`, stratified). This guarantees the numbers
  shown in the app match the notebook exactly, without needing to ship the raw split as a
  separate file.
- `requirements.txt` pins `pandas`, `numpy`, and `scikit-learn` to the versions used to train
  the models, to avoid pickle-compatibility issues. If Streamlit Cloud cannot resolve those exact
  versions in the future, relax the pins slightly and re-run the notebook's Section 9 to
  re-save the model files with the newer versions.
