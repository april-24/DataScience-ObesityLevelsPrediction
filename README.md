# DataScience-ObesityLevelsPrediction

Access link: https://datascience-obesitylevelsprediction-2vpzgfxdbnnw5kcgxfum2l.streamlit.app/

## Folder structure

```
GroupX_RSWY1S2_DataScienceProject/
├── README.md                                    <- you are here
├── notebook/
│   ├── obesity_data_science_project.ipynb        <- completed, fully-executed notebook
│   └── obesity.csv                               <- raw dataset (place here before re-running)
└── streamlit_app/
    ├── app.py                                    <- Streamlit deployment prototype
    ├── requirements.txt
    ├── README.md                                 <- deployment instructions (read this to deploy)
    ├── .streamlit/config.toml
    └── models/                                   <- trained models + artifacts used by the app
        ├── decision_tree_pipeline.pkl
        ├── random_forest_pipeline.pkl
        ├── svm_pipeline.pkl
        ├── knn_pipeline.pkl
        ├── label_encoder.pkl
        ├── feature_metadata.json
        ├── model_comparison.csv
        └── obesity_cleaned.csv
```

## ⚠️ Before you submit

The naming format required by the assignment spec is `GroupX_RSWY1S2_DataScienceProject.zip`.
**Rename this top-level folder / zip** to match your actual group number and tutorial group code
(e.g. `Group3_RSWY1S2_DataScienceProject.zip`) before submitting.

## What's in the notebook

1. **Import Libraries**
2. **Data Understanding** — shape, dtypes, unique values, target distribution
3. **Data Preparation** — missing values, column renaming (readable names), duplicates,
   validity checks, outlier detection (IQR / Z-score / Modified Z-score), BMI feature
   engineering (EDA-only), standardisation demo (min-max / z-score / decimal scaling),
   categorisation demo (equal-width / equal-frequency binning)
4. **Descriptive Data Analysis** — frequency tables for every categorical variable
5. **Exploratory Data Analysis** — histograms, ordered obesity-level countplot, pie chart,
   boxplots, scatterplot, correlation heatmap, OLAP-style pivot tables (roll-up/drill-down),
   categorical-vs-obesity stacked bar charts, numeric-vs-severity correlation
6. **ML Preparation** — train/test split (stratified), `OneHotEncoder(handle_unknown="ignore",
   sparse_output=False)` + `StandardScaler` inside a `ColumnTransformer`, a shared 5-fold
   `StratifiedKFold` cross-validation strategy, and a reusable evaluation helper
7. **Modelling**
   - 7a. Decision Tree — **baseline** (untuned), reports train **and** test performance (Kyra)
   - 7b. Random Forest — `GridSearchCV` tuned, + feature importance (Liping)
   - 7c. SVM — `GridSearchCV` tuned (Wenhsuan)
   - 7d. KNN — `GridSearchCV` tuned, + k-sensitivity curve (Gladys)
8. **Evaluation** — consolidated comparison table, **inner comparison** (train vs test accuracy
   per model = overfitting check), **outer comparison** (model vs model on the test set),
   confusion matrices, ROC curves, weighted AUC
9. **Save Models & Artifacts** — saves everything the Streamlit app needs into `models/`
10. **Conclusion** — summary of findings, limitations, and future improvements

**Actual results from the executed notebook** (your numbers may vary slightly if you re-run,
since GridSearchCV uses the same random_state but training is still somewhat environment
sensitive):

| Model | Test Accuracy | Weighted F1 | Weighted AUC | Overfit Gap (Train − Test) |
|---|---|---|---|---|
| Decision Tree (Baseline) | 93.3% | 0.933 | 0.961 | 6.7% |
| Random Forest | 93.5% | 0.936 | 0.994 | 6.1% |
| **SVM** | **95.7%** | **0.957** | **0.999** | 2.7% |
| KNN | 86.4% | 0.859 | 0.976 | 13.6% |

SVM is the best-performing model overall (highest accuracy/F1/AUC, smallest overfitting gap),
and is set as the default model in the Streamlit app's Prediction tab.

## How to use the notebook

1. Open `notebook/obesity_data_science_project.ipynb` in Google Colab or Jupyter.
2. Make sure `obesity.csv` is in the same folder (or upload it when Colab prompts).
3. Run all cells top to bottom (`Runtime > Run all` in Colab). It takes a few minutes because of
   the `GridSearchCV` hyperparameter tuning in Sections 7b–7d.
4. All required visuals, tables, and outputs are embedded in the notebook already — copy them
   into your written report as needed (screenshots or "Insert image" in Google Docs).
5. Re-running the notebook regenerates a `models/` folder with fresh `.pkl` files — copy that
   into `streamlit_app/models/` if you want the app to reflect a re-run.

## How to run / deploy the Streamlit app

See **`streamlit_app/README.md`** for full instructions. Quick version:

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

To deploy on Streamlit Community Cloud: push `streamlit_app/` to GitHub and connect it at
[share.streamlit.io](https://share.streamlit.io) — full step-by-step instructions are in
`streamlit_app/README.md`.

## Mapping to the marking rubric

- **Model Selection (CLO1):** 4 models incl. one explicit baseline, hyperparameter tuning via
  `GridSearchCV`, k-sensitivity analysis for KNN, feature importance for Random Forest.
- **Data Preprocessing (CLO2):** renaming, missing-value/duplicate checks, 3 outlier-detection
  methods, standardisation demo, binning demo, documented decisions.
- **Descriptive & Exploratory Analysis (CLO2):** frequency tables, univariate/bivariate plots,
  OLAP-style pivot tables, correlation analysis — each with a short interpretation.
- **Graphing & Visualisation (CLO2):** histogram, pie chart, scatterplot, boxplots, heatmap,
  stacked bar charts, ROC curves — all via Matplotlib/Seaborn.
- **Advanced Analytics & Discussion (CLO3):** inner + outer model comparisons, over/underfitting
  discussion, ROC/AUC, and a **functional deployment prototype** (Streamlit app).
- **Report Structure / Presentation (CLO3):** this README + notebook markdown cells map directly
  onto the required report sections (Business Understanding → Conclusion) — use them as your
  outline when writing the Google Docs report.

## Still to do (outside this ZIP — these belong in your Google Docs report, not the code ZIP)

- Write the **Executive Summary**, expand **Business Understanding** with your "increase in
  obesity rate" article/statistics, and expand each notebook interpretation into full report
  prose.
- Add **5+ references (APA 7th edition)**, including 2 academic papers justifying model choice.
- Take screenshots of the notebook's visuals/tables and paste them into the report (code
  snippets are *not* required in the report per the spec).
- Prepare your **30-minute presentation** (all members must participate).
