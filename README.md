# BMDS2003 Obesity Level Classification — Streamlit Cloud

This project predicts `Obesity_Level` using four basic classification models trained on the supplied obesity dataset:

- Decision Tree (baseline)
- Random Forest
- SVM
- KNN

## Reproducible modelling setup

- Target: `NObeyesdad`, renamed to `Obesity_Level`
- Duplicate rows removed: 24
- Cleaned modelling rows: 2,087
- Features used by the models: 16
- Train/test split: **70% / 30%**
- Split method: **stratified**
- Random state: **42**
- Hyperparameter tuning: **none** (basic/default models)
- All four models use the same preprocessing pipeline: numeric standardisation + one-hot encoding for categorical variables.
- BMI and Age_Group are EDA-only variables and are not used as predictors.

## Verified test-set results

| Model | Test Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
|---|---:|---:|---:|---:|
| Decision Tree | 92.03% | 92.04% | 92.03% | 91.99% |
| Random Forest | 93.78% | 94.28% | 93.78% | 93.89% |
| SVM | 91.39% | 91.46% | 91.39% | 91.41% |
| KNN | 81.34% | 81.09% | 81.34% | 79.91% |

All four test accuracies are different. No artificial score adjustment was used.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

1. Put the contents of this folder in a GitHub repository.
2. In Streamlit Community Cloud, create a new app from that repository.
3. Set the main file path to `app.py`.
4. Deploy.

The `models/` folder must remain in the repository because the app loads the four saved pipelines and evaluation artifacts from it.

## App sections

### About
Project objective, CRISP-DM workflow, data summary, model roles and experiment settings.

### Prediction
The user selects **one** model and submits one prediction. Only the selected pipeline is executed for that prediction request.

### Data Exploration
Interactive target filtering, data preview, distributions, scatter plot, box plot and correlation heatmap.

### Model Performance
Saved test-set metrics, model comparison chart, train-vs-test comparison, four confusion matrices and an explicit check that the four accuracy results are distinct.

## Notebook

`obesity_data_science_project.ipynb` contains the data understanding, cleaning, EDA, 70/30 split, preprocessing, training, evaluation, requirement validation and artifact-generation steps.
