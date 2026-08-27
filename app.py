import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

st.set_page_config(
    page_title="Obesity Level Classifier",
    page_icon="📊",
    layout="wide"
)

TARGET = "NObeyesdad"
DEFAULT_DATA = Path(__file__).with_name("obesity.csv")


@st.cache_data
def load_csv(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif DEFAULT_DATA.exists():
        df = pd.read_csv(DEFAULT_DATA)
    else:
        return None

    if TARGET not in df.columns:
        raise ValueError(f"The dataset must contain a '{TARGET}' target column.")
    return df.drop_duplicates().reset_index(drop=True)


def make_preprocessor(X, scale_numeric=False):
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols)
    ]), numeric_cols, categorical_cols


@st.cache_resource
def train_models(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    pre_scaled, _, _ = make_preprocessor(X, scale_numeric=True)
    pre_unscaled, _, _ = make_preprocessor(X, scale_numeric=False)

    models = {
        # Best parameters selected by 3-fold weighted-F1 grid search in the notebook.
        "Decision Tree": Pipeline([
            ("preprocess", pre_unscaled),
            ("model", DecisionTreeClassifier(
                criterion="entropy",
                max_depth=10,
                min_samples_split=5,
                random_state=42
            ))
        ]),
        "Random Forest": Pipeline([
            ("preprocess", pre_unscaled),
            ("model", RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1
            ))
        ]),
        "SVM": Pipeline([
            ("preprocess", pre_scaled),
            ("model", SVC(
                C=10,
                kernel="linear",
                gamma="scale",
                probability=True,
                random_state=42
            ))
        ]),
        "KNN": Pipeline([
            ("preprocess", pre_scaled),
            ("model", KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                p=1
            ))
        ])
    }

    return X, y, models


st.title("📊 Obesity Level Classification")
st.caption("BMDS2003 Data Science — Decision Tree, Random Forest, SVM and KNN")

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader(
        "Upload another CSV (optional)",
        type=["csv"],
        help="The CSV must contain the NObeyesdad target column."
    )

try:
    df = load_csv(uploaded)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if df is None:
    st.warning("No obesity.csv file was found. Upload the dataset using the sidebar.")
    st.stop()

st.sidebar.success(f"{len(df):,} rows loaded")
st.sidebar.write("Target:", TARGET)

X, y, models = train_models(df)

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Exploratory Data Analysis",
    "Model Comparison",
    "Predict Obesity Level"
])

with tab1:
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Features", f"{df.shape[1] - 1}")
    c3.metric("Classes", f"{y.nunique()}")
    c4.metric("Missing cells", f"{int(df.isna().sum().sum())}")

    st.write("### Data preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.write("### Class distribution")
    counts = y.value_counts().rename_axis("Obesity Level").reset_index(name="Count")
    st.bar_chart(counts.set_index("Obesity Level"))

    st.info(
        "The prototype treats NObeyesdad as the output label. "
        "It is a classification tool for the assignment and should not be treated as a clinical diagnosis."
    )

with tab2:
    st.subheader("Exploratory Data Analysis")

    st.write("### Target distribution")
    counts = y.value_counts()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Obesity level")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Obesity Levels")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig, clear_figure=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    selected_num = st.selectbox("Choose a numeric feature", numeric_cols, index=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df[selected_num], bins=20)
    ax.set_title(f"Distribution of {selected_num}")
    ax.set_xlabel(selected_num)
    ax.set_ylabel("Frequency")
    st.pyplot(fig, clear_figure=True)

    st.write("### Summary statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

with tab3:
    st.subheader("Model Comparison")

    # Use a deterministic stratified holdout consistent with the notebook.
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    evaluation = []
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        evaluation.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision (weighted)": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall (weighted)": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1 (weighted)": f1_score(y_test, pred, average="weighted", zero_division=0)
        })

    result_df = pd.DataFrame(evaluation).sort_values("F1 (weighted)", ascending=False)

    st.dataframe(
        result_df.style.format({
            "Accuracy": "{:.4f}",
            "Precision (weighted)": "{:.4f}",
            "Recall (weighted)": "{:.4f}",
            "F1 (weighted)": "{:.4f}"
        }),
        use_container_width=True
    )

    best_name = result_df.iloc[0]["Model"]
    st.success(f"Best model on this holdout by weighted F1: {best_name}")

    st.write("### Metric comparison")
    chart_df = result_df.set_index("Model")[[
        "Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 (weighted)"
    ]]
    st.bar_chart(chart_df)

    selected_model = st.selectbox("Inspect confusion matrix", result_df["Model"].tolist())
    labels = sorted(y.unique())
    fig, ax = plt.subplots(figsize=(9, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, predictions[selected_model], labels=labels),
        display_labels=labels
    )
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"Confusion Matrix — {selected_model}")
    st.pyplot(fig, clear_figure=True)

    st.write("### Classification report")
    report = classification_report(
        y_test, predictions[selected_model], output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T, use_container_width=True)

with tab4:
    st.subheader("Single-Record Prediction")
    st.write("Enter a person's characteristics to obtain a predicted obesity-level class.")

    input_values = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            series = X[col]
            default = float(series.median())
            min_value = float(series.min())
            max_value = float(series.max())
            step = 0.1 if not float(default).is_integer() else 1.0
            input_values[col] = st.number_input(
                col,
                min_value=min_value,
                max_value=max_value,
                value=default,
                step=step
            )
        else:
            choices = sorted(X[col].dropna().astype(str).unique().tolist())
            input_values[col] = st.selectbox(col, choices)

    model_choice = st.selectbox(
        "Prediction model",
        list(models.keys()),
        index=0
    )

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([input_values])
        model = models[model_choice]
        model.fit(X, y)
        prediction = model.predict(row)[0]

        st.success(f"Predicted obesity level: **{prediction}**")

        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(row)[0]
                prob_df = pd.DataFrame({
                    "Obesity level": model.classes_,
                    "Probability": probabilities
                }).sort_values("Probability", ascending=False)
                st.write("### Class probabilities")
                st.dataframe(
                    prob_df.style.format({"Probability": "{:.2%}"}),
                    use_container_width=True
                )
            except Exception:
                pass

st.divider()
st.caption(
    "Assignment prototype: results depend on the supplied dataset and the fixed modelling configuration. "
    "For academic submission, use the notebook outputs and report discussion together."
)
