# DataScience-ObesityLevelsPrediction

**Access link: [https://datascience-obesitylevelsprediction-2vpzgfxdbnnw5kcgxfum2l.streamlit.app/](https://ds-obesitylevels-h8tbkwcgggks8jmeeqebpa.streamlit.app/)**

# BMDS2003 Data Science — Group Project
## Estimation of Obesity Levels Based on Eating Habits and Physical Condition

CRISP-DM implementation + Streamlit deployment prototype for the group assignment.

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

**Actual results from the executed notebook:**

| Model | Train Acc. (in-sample) | Train Acc. (5-fold CV) | Test Accuracy | Weighted F1 | Macro F1 | AUC | Overfit Gap (CV − Test) |
|---|---|---|---|---|---|---|---|
| Decision Tree (Baseline) | 1.0000 | 0.9173 | 0.9330 | 0.9326 | 0.9305 | 0.9610 | −0.0157 |
| Random Forest | 0.9964 | 0.9383 | 0.9354 | 0.9365 | 0.9337 | 0.9938 | +0.0029 |
| **SVM** | 0.9838 | **0.9641** | **0.9569** | **0.9571** | **0.9562** | **0.9992** | +0.0071 |
| KNN | 1.0000 | 0.8766 | 0.8636 | 0.8585 | 0.8528 | 0.9758 | +0.0129 |

SVM is the best-performing model overall and is set as the default model in the Streamlit app.

**A note on the two "training accuracy" columns above** (this matters — read before writing your
report's Evaluation section): `Train_Accuracy_InSample` is the model scored on the *same rows it
was fit on*. It reaches exactly **1.0000** for the Decision Tree and KNN — this is mathematically
expected, not a bug: the Decision Tree uses default (unconstrained) hyperparameters, so it keeps
splitting until every training leaf is pure, and `GridSearchCV` selected `weights="distance"` for
KNN, which means every training point's nearest neighbour during scoring is itself at distance 0.
Neither is a meaningful "how well did the model learn" number, because both are trivially forced
toward 1.0 by construction.

The column that should actually be used to judge over/under-fitting is `Train_Accuracy_CV` — the
mean accuracy across 5 cross-validation folds, where each fold's model is scored on training rows
it did **not** see during that fold's fit. Once measured this way, the "Overfit Gap" for every
model is small (a few percentage points at most, and even *negative* for the baseline), meaning
**none of the four models seriously overfit** — the in-sample 1.0000 numbers were simply the wrong
statistic to compare against test accuracy. The notebook (Section 7a's markdown, and Section 8.2)
walks through this distinction in full, with both numbers reported side by side for transparency.

**On why Accuracy / Precision / Recall / F1 can look similar:** Weighted Recall is *mathematically
identical* to Accuracy for any single-label multiclass classifier (a provable identity, not a
coincidence — see the notebook markdown for the derivation) — so those two will always match
exactly, on any dataset. Weighted Precision and F1 are not identities and genuinely can diverge;
the notebook and app now also report **Macro**-averaged Precision/Recall/F1 (each class counted
equally regardless of size) alongside the Weighted versions specifically to make this divergence
visible — see how much further KNN's macro scores fall below its weighted scores in the table
above, evidence that its errors concentrate in specific harder classes rather than spreading evenly.

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

