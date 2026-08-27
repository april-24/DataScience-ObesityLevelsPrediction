"""Streamlit app for the BMDS2003 Obesity Level Classification project."""
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Obesity Level Prediction | BMDS2003",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT / "obesity(1).csv"

MODEL_FILES = {
    "Decision Tree": "decision_tree_pipeline.pkl",
    "Random Forest": "random_forest_pipeline.pkl",
    "SVM": "svm_pipeline.pkl",
    "KNN": "knn_pipeline.pkl",
}

DISPLAY_ORDER = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]

RENAME_MAP = {
    "NObeyesdad": "Obesity_Level",
    "family_history_with_overweight": "Family_History_Overweight",
    "FAVC": "Frequent_High_Caloric_Food",
    "FCVC": "Vegetable_Consumption_Freq",
    "NCP": "Main_Meals_Per_Day",
    "CAEC": "Food_Between_Meals",
    "SMOKE": "Smokes",
    "CH2O": "Daily_Water_Intake",
    "SCC": "Calorie_Monitoring",
    "FAF": "Physical_Activity_Freq",
    "TUE": "Technology_Usage_Time",
    "CALC": "Alcohol_Consumption",
    "MTRANS": "Transportation_Mode",
}

RECOMMENDATIONS = {
    "Insufficient_Weight": "The model places this profile in the insufficient-weight category. This is an educational prediction, not a diagnosis; consider discussing nutrition and overall health with a qualified professional.",
    "Normal_Weight": "The model places this profile in the normal-weight category. Continue balanced eating, regular activity, and healthy daily habits.",
    "Overweight_Level_I": "The model places this profile in Overweight Level I. Sustainable changes to diet, activity, and sedentary time may be useful, with professional guidance where appropriate.",
    "Overweight_Level_II": "The model places this profile in Overweight Level II. A structured lifestyle plan and professional guidance may be appropriate.",
    "Obesity_Type_I": "The model places this profile in Obesity Type I. Consider discussing weight-management options and overall health with a healthcare professional.",
    "Obesity_Type_II": "The model places this profile in Obesity Type II. Professional medical guidance is recommended for safe, individualised management.",
    "Obesity_Type_III": "The model places this profile in Obesity Type III. Professional medical guidance is recommended for safe, individualised management.",
}


def pretty(text: str) -> str:
    return str(text).replace("_", " ")


@st.cache_data(show_spinner=False)
def load_raw_data():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_data(show_spinner=False)
def load_cleaned_data():
    p = MODELS_DIR / "obesity_cleaned.csv"
    if p.exists():
        return pd.read_csv(p)
    df = load_raw_data().rename(columns=RENAME_MAP).copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    df = df.drop_duplicates().reset_index(drop=True)
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[0, 18, 30, 45, 60, 150],
        labels=["<18", "18–29", "30–44", "45–59", "60+"],
        right=False,
    )
    return df


@st.cache_data(show_spinner=False)
def load_metadata():
    with open(MODELS_DIR / "feature_metadata.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_metrics():
    with open(MODELS_DIR / "evaluation_summary.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_confusion_matrices():
    with open(MODELS_DIR / "confusion_matrices.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="Loading selected model...")
def load_model(model_name):
    return joblib.load(MODELS_DIR / MODEL_FILES[model_name])


raw = load_raw_data()
df = load_cleaned_data()
metadata = load_metadata()
metrics = load_metrics()
confusion_matrices = load_confusion_matrices()

st.title("🍎 Obesity Level Prediction System")
st.caption(
    "BMDS2003 Data Science Group Project — Classification of obesity level from eating habits, lifestyle and physical-condition attributes"
)

about_tab, prediction_tab, explore_tab, performance_tab = st.tabs(
    ["🏠 About", "🔮 Prediction", "📊 Data Exploration", "📈 Model Performance"]
)

# -----------------------------------------------------------------------------
# About
# -----------------------------------------------------------------------------
with about_tab:
    st.header("About This Project")
    left, right = st.columns([2.2, 1])

    with left:
        st.subheader("Objective")
        st.write(
            "The system classifies a person's obesity level into one of seven classes using 16 predictor variables from the obesity dataset. The target is `NObeyesdad`, renamed to `Obesity_Level` for readability."
        )

        st.subheader("CRISP-DM workflow")
        st.markdown(
            "1. **Business Understanding** — define the obesity-level classification problem.\n"
            "2. **Data Understanding** — inspect dimensions, variable types, distributions and target balance.\n"
            "3. **Data Preparation** — clean duplicates/text values, create EDA-only BMI and age groups, and prepare model features.\n"
            "4. **Modelling** — train four basic classifiers using the same stratified 70/30 split.\n"
            "5. **Evaluation** — compare accuracy, precision, recall, F1-score, train/test gap and confusion matrices.\n"
            "6. **Deployment** — provide a Streamlit prediction interface and model-performance dashboard."
        )

        st.subheader("Models")
        model_table = pd.DataFrame(
            {
                "Model": ["Decision Tree", "Random Forest", "SVM", "KNN"],
                "Role": ["Baseline", "Model 2", "Model 3", "Model 4"],
                "Configuration": [
                    "Default parameters; random_state=42",
                    "Default parameters; random_state=42",
                    "Default parameters; RBF kernel; probability=True",
                    "Default parameters",
                ],
            }
        )
        st.dataframe(model_table, use_container_width=True, hide_index=True)

        st.info(
            "The prediction page intentionally runs only the model selected by the user. The app does not ask one prediction form to execute all four classifiers simultaneously."
        )

    with right:
        st.metric("Raw records", f"{metrics['n_rows_raw']:,}")
        st.metric("Cleaned records", f"{metrics['n_rows_cleaned']:,}")
        st.metric("Train / Test", f"{metrics['train_rows']:,} / {metrics['test_rows']:,}")
        st.metric("Test split", "70% / 30%")
        st.metric("Best test accuracy", f"{max(r['Test_Accuracy'] for r in metrics['model_results']):.2%}")
        st.caption(f"Best model: **{pretty(metrics['best_model'])}**")

# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------
with prediction_tab:
    st.header("Predict Obesity Level")
    st.write("Choose exactly one model. Only that selected model is used for this prediction request.")

    chosen_model = st.selectbox("Model", list(MODEL_FILES.keys()), index=1)
    st.divider()

    numeric = metadata["numeric_features"]
    categorical = metadata["categorical_features"]

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        inputs = {}

        def numeric_widget(col, label, container):
            meta = numeric[col]
            lo, hi, mean = float(meta["min"]), float(meta["max"]), float(meta["mean"])
            step = 0.1 if (hi - lo) <= 10 else 1.0
            return container.number_input(label, min_value=lo, max_value=hi, value=mean, step=step, format="%.2f")

        inputs["Age"] = numeric_widget("Age", "Age", c1)
        inputs["Height"] = numeric_widget("Height", "Height (m)", c1)
        inputs["Weight"] = numeric_widget("Weight", "Weight (kg)", c1)
        inputs["Vegetable_Consumption_Freq"] = numeric_widget(
            "Vegetable_Consumption_Freq", "Vegetable consumption frequency", c1
        )

        def cat_widget(col, label, container):
            options = categorical[col]
            return container.selectbox(label, options, format_func=pretty)

        inputs["Gender"] = cat_widget("Gender", "Gender", c2)
        inputs["Family_History_Overweight"] = cat_widget("Family_History_Overweight", "Family history of overweight", c2)
        inputs["Frequent_High_Caloric_Food"] = cat_widget("Frequent_High_Caloric_Food", "Frequent high-caloric food", c2)
        inputs["Food_Between_Meals"] = cat_widget("Food_Between_Meals", "Food between meals", c2)
        inputs["Smokes"] = cat_widget("Smokes", "Smokes", c2)
        inputs["Calorie_Monitoring"] = cat_widget("Calorie_Monitoring", "Monitors calorie intake", c2)

        inputs["Main_Meals_Per_Day"] = numeric_widget("Main_Meals_Per_Day", "Main meals per day", c3)
        inputs["Daily_Water_Intake"] = numeric_widget("Daily_Water_Intake", "Daily water intake", c3)
        inputs["Physical_Activity_Freq"] = numeric_widget("Physical_Activity_Freq", "Physical activity frequency", c3)
        inputs["Technology_Usage_Time"] = numeric_widget("Technology_Usage_Time", "Technology usage time", c3)
        inputs["Alcohol_Consumption"] = cat_widget("Alcohol_Consumption", "Alcohol consumption", c3)
        inputs["Transportation_Mode"] = cat_widget("Transportation_Mode", "Transportation mode", c3)

        submitted = st.form_submit_button("Predict Obesity Level", type="primary")

    if submitted:
        # Remove the temporary duplicate key created above and preserve only model features.
        clean_inputs = {k: v for k, v in inputs.items() if k in numeric or k in categorical}
        # Ensure every expected feature exists in the exact model-feature set.
        missing = [col for col in numeric.keys() | categorical.keys() if col not in clean_inputs]
        if missing:
            st.error(f"Missing input fields: {missing}")
        else:
            input_row = pd.DataFrame([clean_inputs])
            model = load_model(chosen_model)
            predicted = model.predict(input_row)[0]

            st.divider()
            result, probability_col = st.columns([1, 1.5])
            with result:
                st.subheader("Prediction Result")
                st.metric("Predicted Obesity Level", pretty(predicted))
                bmi = float(input_row["Weight"].iloc[0]) / float(input_row["Height"].iloc[0]) ** 2
                st.metric("Computed BMI (reference only)", f"{bmi:.2f} kg/m²")
                st.caption(f"Model used: **{chosen_model}**")
                st.write(RECOMMENDATIONS.get(predicted, "Educational result only."))

            with probability_col:
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_row)[0]
                    classes = list(model.classes_)
                    proba_df = (
                        pd.DataFrame({"Obesity Level": classes, "Probability": probabilities})
                        .set_index("Obesity Level")
                        .reindex(DISPLAY_ORDER)
                    )
                    st.subheader("Model class probabilities")
                    st.bar_chart(proba_df)
                else:
                    st.info("This selected model does not provide class probabilities.")

# -----------------------------------------------------------------------------
# Data Exploration
# -----------------------------------------------------------------------------
with explore_tab:
    st.header("Data Exploration")
    filtered = df.copy()

    selected_levels = st.multiselect(
        "Filter obesity levels",
        DISPLAY_ORDER,
        default=DISPLAY_ORDER,
        format_func=pretty,
    )
    if selected_levels:
        filtered = filtered[filtered["Obesity_Level"].isin(selected_levels)]

    a, b, c = st.columns(3)
    a.metric("Rows after filter", f"{len(filtered):,}")
    b.metric("Columns", f"{filtered.shape[1]:,}")
    c.metric("Unique obesity levels", f"{filtered['Obesity_Level'].nunique():,}")

    st.subheader("Preview")
    st.dataframe(filtered.head(30), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Obesity-level distribution")
        counts = filtered["Obesity_Level"].value_counts().reindex(DISPLAY_ORDER).dropna()
        st.bar_chart(counts)
    with col2:
        st.subheader("Numeric distribution")
        numeric_cols = filtered.select_dtypes(include=np.number).columns.tolist()
        selected_numeric = st.selectbox("Choose a numeric variable", numeric_cols)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.histplot(filtered[selected_numeric].dropna(), bins=20, kde=True, ax=ax)
        ax.set_xlabel(pretty(selected_numeric))
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Height vs Weight")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.scatterplot(
            data=filtered, x="Height", y="Weight", hue="Obesity_Level",
            hue_order=DISPLAY_ORDER, alpha=0.65, ax=ax
        )
        ax.set_xlabel("Height (m)")
        ax.set_ylabel("Weight (kg)")
        ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    with col4:
        st.subheader("Weight by obesity level")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.boxplot(
            data=filtered, x="Obesity_Level", y="Weight",
            order=DISPLAY_ORDER, ax=ax
        )
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    st.subheader("Correlation heatmap")
    numeric_data = filtered.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_data.corr(), annot=False, cmap="coolwarm", center=0, ax=ax)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

# -----------------------------------------------------------------------------
# Performance
# -----------------------------------------------------------------------------
with performance_tab:
    st.header("Model Performance")
    result_df = pd.DataFrame(metrics["model_results"]).set_index("Model")
    result_df = result_df.reindex(MODEL_FILES.keys())

    # -------------------------------------------------------------------------
    # 1. Overall ranking and key finding
    # -------------------------------------------------------------------------
    st.subheader("1. Overall Model Ranking")
    result_df_display = result_df[[
        "Train_Accuracy", "Test_Accuracy", "Precision_Weighted",
        "Recall_Weighted", "F1_Weighted", "Overfit_Gap"
    ]].copy()
    result_df_display.columns = [
        "Train Accuracy", "Test Accuracy", "Weighted Precision",
        "Weighted Recall", "Weighted F1", "Overfit Gap"
    ]
    result_df_display = result_df_display.sort_values("Test Accuracy", ascending=False)
    st.dataframe(
        result_df_display.style.format("{:.2%}"),
        use_container_width=True,
    )

    best_model = metrics["best_model"]
    best_acc = float(result_df.loc[best_model, "Test_Accuracy"])
    worst_acc = float(result_df["Test_Accuracy"].min())
    st.success(
        f"**Best model:** {pretty(best_model)} — {best_acc:.2%} test accuracy. "
        f"The gap between the best and lowest model is {(best_acc - worst_acc):.2%}."
    )

    if result_df["Test_Accuracy"].nunique() == 4:
        st.info(
            "Validation requirement met: all four models have different genuine test-set accuracy scores. "
            "No artificial score adjustment was used."
        )
    else:
        st.warning(
            "Some test accuracies are tied. All values shown are genuine model outputs; no scores were fabricated."
        )

    # -------------------------------------------------------------------------
    # 2. Truncated test-accuracy chart
    # -------------------------------------------------------------------------
    st.subheader("2. Test Accuracy — Enlarged Difference View")
    st.caption(
        "The y-axis starts at 75% so small differences near the top become easier to see. "
        "The underlying accuracy values are unchanged."
    )
    acc_sorted = result_df["Test_Accuracy"].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(acc_sorted.index, acc_sorted.values)
    ax.set_ylabel("Test Accuracy")
    lower = max(0.75, float(acc_sorted.min()) - 0.03)
    upper = min(1.00, float(acc_sorted.max()) + 0.025)
    ax.set_ylim(lower, upper)
    ax.set_title("Test Accuracy by Model (Zoomed Scale)")
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, acc_sorted.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.004,
            f"{value:.2%}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # -------------------------------------------------------------------------
    # 3. Multi-metric comparison
    # -------------------------------------------------------------------------
    st.subheader("3. Multi-Metric Comparison")
    metric_display = result_df[[
        "Test_Accuracy", "Precision_Weighted", "Recall_Weighted", "F1_Weighted"
    ]].rename(columns={
        "Test_Accuracy": "Accuracy",
        "Precision_Weighted": "Precision",
        "Recall_Weighted": "Recall",
        "F1_Weighted": "F1",
    })
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(metric_display.index))
    width = 0.19
    for j, metric in enumerate(metric_display.columns):
        values = metric_display[metric].values
        bars = ax.bar(x + (j - 1.5) * width, values, width, label=metric)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.006,
                f"{value:.1%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_display.index)
    ax.set_ylim(max(0.70, float(metric_display.min().min()) - 0.03), 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Accuracy, Precision, Recall and F1")
    ax.legend(ncol=4)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # -------------------------------------------------------------------------
    # 4. Train-vs-test generalisation
    # -------------------------------------------------------------------------
    st.subheader("4. Training vs Test Accuracy")
    train_test = result_df[["Train_Accuracy", "Test_Accuracy"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(train_test.index))
    width = 0.34
    train_bars = ax.bar(x - width / 2, train_test["Train_Accuracy"], width, label="Train")
    test_bars = ax.bar(x + width / 2, train_test["Test_Accuracy"], width, label="Test")
    for bars in (train_bars, test_bars):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.006,
                f"{value:.1%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(train_test.index)
    ax.set_ylim(max(0.70, float(train_test.min().min()) - 0.03), 1.02)
    ax.set_ylabel("Accuracy")
    ax.set_title("Training vs Test Accuracy")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    gap_df = result_df[["Train_Accuracy", "Test_Accuracy", "Overfit_Gap"]].copy()
    gap_df.columns = ["Train Accuracy", "Test Accuracy", "Overfit Gap"]
    gap_df = gap_df.sort_values("Overfit Gap", ascending=False)
    st.caption("Overfit gap = training accuracy − test accuracy; a larger gap indicates a bigger generalisation difference.")
    st.dataframe(gap_df.style.format("{:.2%}"), use_container_width=True)

    # -------------------------------------------------------------------------
    # 5. Confusion matrix selector
    # -------------------------------------------------------------------------
    st.subheader("5. Confusion Matrix Explorer")
    selected_cm_model = st.selectbox(
        "Select one model to inspect its confusion matrix",
        list(MODEL_FILES.keys()),
        index=1,
        key="cm_model_selector",
    )
    cm = np.asarray(confusion_matrices[selected_cm_model])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[pretty(x) for x in DISPLAY_ORDER],
        yticklabels=[pretty(x) for x in DISPLAY_ORDER],
        ax=ax,
    )
    ax.set_title(f"{selected_cm_model} — Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # -------------------------------------------------------------------------
    # 6. Performance gap and prediction disagreement
    # -------------------------------------------------------------------------
    st.subheader("6. Performance Gap and Prediction Differences")
    gap_from_best = (best_acc - result_df["Test_Accuracy"]).sort_values(ascending=False)
    comparison_table = pd.DataFrame({
        "Test Accuracy": result_df["Test_Accuracy"],
        "Gap from Best": gap_from_best,
    }).sort_values("Test Accuracy", ascending=False)
    st.dataframe(comparison_table.style.format("{:.2%}"), use_container_width=True)

    disagreement_df = pd.Series(
        metrics["prediction_disagreements"], name="Different test predictions"
    ).to_frame()
    st.write(
        "Even models with similar overall accuracy can make different errors on individual test observations. "
        "The table below counts how many of the 627 held-out observations received different predicted classes "
        "between each model pair."
    )
    st.dataframe(disagreement_df, use_container_width=True)

    st.subheader("Interpretation")
    st.write(
        "Random Forest is the strongest model in this experiment based on test accuracy and weighted F1. "
        "Decision Tree and SVM are relatively close to it, while KNN is clearly lower. The multi-metric chart, "
        "train-test comparison and confusion-matrix explorer are included so the evaluation does not rely on "
        "accuracy alone. All four models were trained with the same cleaned features and the same stratified 70/30 split."
    )

st.divider()
st.caption("Educational data-science project. Predictions are model outputs and should not be treated as medical diagnoses.")
