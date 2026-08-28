"""
BMDS2003 Data Science — Obesity Level Prediction App
======================================================
A Streamlit deployment prototype for the group project
"Estimation of Obesity Levels Based on Eating Habits and Physical Condition".

Run locally with:
    streamlit run app.py

Required files:
    app.py
    requirements.txt
    models/
        decision_tree_pipeline.pkl
        random_forest_pipeline.pkl
        svm_pipeline.pkl
        knn_pipeline.pkl
        label_encoder.pkl
        feature_metadata.json
        obesity_cleaned.csv
        model_comparison.csv
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Obesity Level Predictor | BMDS2003",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

MODELS_DIR = Path(__file__).parent / "models"

MODEL_FILES = {
    "Decision Tree (Baseline)": "decision_tree_pipeline.pkl",
    "Random Forest": "random_forest_pipeline.pkl",
    "SVM": "svm_pipeline.pkl",
    "KNN": "knn_pipeline.pkl",
}

RECOMMENDATIONS = {
    "Insufficient_Weight": (
        "Your inputs suggest an underweight profile. Consider a structured nutrition plan "
        "that gradually increases calorie-dense, nutritious foods, and consult a healthcare "
        "professional to rule out underlying causes."
    ),
    "Normal_Weight": (
        "Your inputs suggest a healthy weight range. Keep up balanced meals, regular "
        "physical activity, and adequate water intake to maintain this."
    ),
    "Overweight_Level_I": (
        "Your inputs suggest early-stage overweight. Small, sustainable changes — more "
        "vegetables, less frequent high-caloric snacking, and 2-3 extra active sessions a "
        "week — can meaningfully reduce risk of progressing further."
    ),
    "Overweight_Level_II": (
        "Your inputs suggest overweight. A more structured plan combining diet adjustment, "
        "increased physical activity frequency, and reduced sedentary technology time is "
        "recommended, ideally with professional guidance."
    ),
    "Obesity_Type_I": (
        "Your inputs suggest Class I obesity. We recommend consulting a healthcare provider "
        "or dietitian to design a supervised weight-management plan, alongside gradual "
        "increases in physical activity."
    ),
    "Obesity_Type_II": (
        "Your inputs suggest Class II obesity. Professional medical guidance is strongly "
        "recommended to design a safe, supervised intervention plan."
    ),
    "Obesity_Type_III": (
        "Your inputs suggest Class III (severe) obesity. Please consult a healthcare "
        "professional promptly to discuss a comprehensive, medically supervised management "
        "plan."
    ),
}


# ============================================================
# CONSISTENT OBESITY LEVEL ORDER
# ============================================================

OBESITY_COLORS = {
    "Insufficient_Weight": "#4C78A8",
    "Normal_Weight": "#59A14F",
    "Overweight_Level_I": "#F2CF5B",
    "Overweight_Level_II": "#F28E2B",
    "Obesity_Type_I": "#E15759",
    "Obesity_Type_II": "#B279A2",
    "Obesity_Type_III": "#7F3C8D",
}


CHART_TEMPLATE = "plotly_white"


# ============================================================
# HELPER FUNCTIONS — CHART STYLING
# ============================================================

def style_chart(fig, height=500):
    """
    Apply a consistent visual style to Plotly charts.
    """

    fig.update_layout(
        template=CHART_TEMPLATE,
        height=height,
        margin=dict(
            l=60,
            r=30,
            t=75,
            b=80,
        ),
        font=dict(
            size=13,
        ),
        hoverlabel=dict(
            font_size=13,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
    )

    return fig


def display_chart(fig):
    """
    Display an interactive Plotly chart with Streamlit.
    """

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "responsive": True,
        },
    )


# ============================================================
# CACHED LOADERS
# ============================================================

@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    models = {}

    for name, filename in MODEL_FILES.items():
        path = MODELS_DIR / filename

        if path.exists():
            models[name] = joblib.load(path)

    label_encoder_path = MODELS_DIR / "label_encoder.pkl"

    if not label_encoder_path.exists():
        raise FileNotFoundError(
            f"Missing label encoder: {label_encoder_path}"
        )

    label_encoder = joblib.load(label_encoder_path)

    return models, label_encoder


@st.cache_resource(show_spinner=False)
def load_metadata():

    metadata_path = MODELS_DIR / "feature_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata file: {metadata_path}"
        )

    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading dataset...")
def load_cleaned_data():

    data_path = MODELS_DIR / "obesity_cleaned.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing cleaned dataset: {data_path}"
        )

    return pd.read_csv(data_path)


@st.cache_data(show_spinner=False)
def load_comparison_table():

    comparison_path = MODELS_DIR / "model_comparison.csv"

    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Missing model comparison file: {comparison_path}"
        )

    return pd.read_csv(
        comparison_path,
        index_col=0,
    )


@st.cache_data(show_spinner="Rebuilding the held-out test split for evaluation...")
def rebuild_test_split(_metadata):
    """
    Recreate the exact same train/test split used in the notebook.

    The split uses:
        test_size = 0.20
        random_state = 42
        stratify = target

    This allows the Model Performance tab to evaluate the saved
    pipelines on the same held-out test set.
    """

    df = load_cleaned_data()

    exclude_cols = [
        "Obesity_Level",
        "BMI",
        "Age_Group",
    ]

    X = df.drop(
        columns=[
            c for c in exclude_cols
            if c in df.columns
        ]
    )

    y = df["Obesity_Level"]

    target_classes = _metadata["target_classes"]

    class_to_int = {
        c: i
        for i, c in enumerate(target_classes)
    }

    y_encoded = y.map(class_to_int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded,
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# LOAD EVERYTHING
# ============================================================

models, label_encoder = load_models()
metadata = load_metadata()

obesity_order = metadata["obesity_order"]

best_model_name = metadata.get(
    "best_model",
    "Random Forest",
)


# ============================================================
# HEADER
# ============================================================

st.title("🍎 Obesity Level Prediction System")

st.caption(
    "BMDS2003 Data Science Group Project — Estimation of Obesity Levels Based on "
    "Eating Habits and Physical Condition (CRISP-DM)"
)


# ============================================================
# TABS
# ============================================================

tab_about, tab_predict, tab_explore, tab_performance = st.tabs(
    [
        "🏠 About",
        "🔮 Prediction",
        "📊 Data Exploration",
        "📈 Model Performance",
    ]
)


# ============================================================
# TAB 1 — ABOUT
# ============================================================

with tab_about:

    st.header("About This Project")

    col1, col2 = st.columns([2, 1])

    # --------------------------------------------------------
    # BUSINESS UNDERSTANDING
    # --------------------------------------------------------

    with col1:

        st.subheader("Business Understanding")

        st.markdown(
            """
Obesity is a growing public-health concern linked to diabetes, cardiovascular disease, and
reduced quality of life. Traditional diagnosis relies on clinical BMI measurement, which
requires an in-person visit.

This project explores whether **everyday lifestyle and eating habits alone can predict a
person's obesity category**, enabling:

- **Self-assessment tools** that flag risk before a clinical visit is needed.
- **Public-health screening** at scale (schools, workplaces, community programmes).
- **Targeted lifestyle recommendations** that highlight the *actionable* factors such as
  diet, activity, and transportation habits.

**Objective:** classify an individual into one of seven clinically-defined obesity levels using
16 lifestyle, dietary, and physical-condition attributes.
"""
        )

        st.subheader("Dataset")

        st.markdown(
            """
- **Source:** UCI Machine Learning Repository — *Estimation of Obesity Levels Based on Eating Habits and Physical Condition*.
- **Records:** 2,111 respondents from Mexico, Peru, and Colombia.
- **Variables:** 17 columns — 16 features + target.
- **Target:** `Obesity_Level` — 7 classes ranging from `Insufficient_Weight` to `Obesity_Type_III`.
"""
        )

        st.subheader("CRISP-DM Workflow")

        st.markdown(
            """
1. **Business Understanding** — define the prediction problem and its real-world value.
2. **Data Understanding** — profile the dataset's structure, types, and distributions.
3. **Data Preparation** — clean the data, check outliers, engineer BMI for analysis, and
   demonstrate standardisation and binning techniques.
4. **Modelling** — train 4 classifiers: a Decision Tree baseline, plus tuned Random Forest,
   SVM, and KNN models, using 5-fold cross-validation.
5. **Evaluation** — compare accuracy, precision, recall, F1, AUC, confusion matrices,
   classification reports, and train-vs-test performance.
6. **Deployment** — this Streamlit application allows live prediction, data exploration,
   and model-performance inspection.
"""
        )

    # --------------------------------------------------------
    # TEAM / BEST MODEL
    # --------------------------------------------------------

    with col2:

        st.subheader("Team")

        st.markdown(
            """
| Member | Model Owned |
|---|---|
| Kyra | Decision Tree (Baseline) |
| Liping | Random Forest |
| Wenhsuan | Support Vector Machine |
| Gladys | K-Nearest Neighbours |
"""
        )

        st.subheader("Best Model")

        st.success(
            f"**{best_model_name}** achieved the highest test-set accuracy."
        )

        comparison_preview = load_comparison_table()

        if (
            best_model_name in comparison_preview.index
            and "Test_Accuracy" in comparison_preview.columns
        ):

            st.metric(
                "Best Test Accuracy",
                f"{comparison_preview.loc[best_model_name, 'Test_Accuracy']:.1%}",
            )

        st.info(
            "Use the **Prediction** tab to try the model yourself, the **Data Exploration** "
            "tab to browse the dataset, and the **Model Performance** tab for full evaluation "
            "metrics."
        )


# ============================================================
# TAB 2 — PREDICTION
# ============================================================

with tab_predict:

    st.header("🔮 Predict an Obesity Level")

    st.caption(
        "Fill in the fields below and choose a model to generate a live prediction. "
        "This uses the exact preprocessing + trained classifier pipeline saved from the notebook."
    )

    # --------------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------------

    available_models = [
        m for m in MODEL_FILES
        if m in models
    ]

    if not available_models:

        st.error(
            "No trained models were found in the models/ folder."
        )

        st.stop()

    default_index = (
        available_models.index(best_model_name)
        if best_model_name in available_models
        else 0
    )

    st.markdown("**Choose a model for prediction**")

    chosen_model_name = st.segmented_control(
        "Choose a model for prediction",
        available_models,
        default=available_models[default_index],
        label_visibility="collapsed",
    )

    if chosen_model_name is None:
        chosen_model_name = available_models[default_index]

    if chosen_model_name == best_model_name:

        st.caption(
            f"Using **{chosen_model_name}** ⭐ Best test-set model"
        )

    else:

        st.caption(
            f"Using **{chosen_model_name}**"
        )

    num_meta = metadata["numeric_features"]
    cat_meta = metadata["categorical_features"]

    # Logical ordering for frequency variables
    FREQUENCY_ORDER = [
        "no",
        "Sometimes",
        "Frequently",
        "Always",
    ]

    def freq_label(x):
        return "Never" if x == "no" else x

    def yes_no_label(x):
        return "✅ Yes" if x == "yes" else "❌ No"

    # --------------------------------------------------------
    # INPUT FORM
    # --------------------------------------------------------

    with st.form("prediction_form"):

        st.subheader("Personal & Physical Attributes")

        c1, c2, c3 = st.columns(3)

        with c1:

            st.markdown("**Gender**")

            gender = st.radio(
                "Gender",
                cat_meta["Gender"],
                horizontal=True,
                label_visibility="collapsed",
                format_func=lambda x:
                    "👩 Female"
                    if x == "Female"
                    else "👨 Male",
            )

            age = st.number_input(
                "Age (years)",
                min_value=float(num_meta["Age"]["min"]),
                max_value=100.0,
                value=round(
                    num_meta["Age"]["mean"],
                    1,
                ),
                step=1.0,
            )

        with c2:

            height = st.number_input(
                "Height (m)",
                min_value=float(num_meta["Height"]["min"]),
                max_value=float(num_meta["Height"]["max"]) + 0.3,
                value=round(
                    num_meta["Height"]["mean"],
                    2,
                ),
                step=0.01,
                format="%.2f",
            )

        with c3:

            weight = st.number_input(
                "Weight (kg)",
                min_value=float(num_meta["Weight"]["min"]),
                max_value=float(num_meta["Weight"]["max"]) + 50.0,
                value=round(
                    num_meta["Weight"]["mean"],
                    1,
                ),
                step=1.0,
            )

        # ----------------------------------------------------
        # EATING HABITS
        # ----------------------------------------------------

        st.subheader("Eating Habits")

        c4, c5, c6 = st.columns(3)

        with c4:

            st.markdown(
                "**Family history of overweight?**"
            )

            family_history = st.radio(
                "Family history of overweight?",
                cat_meta["Family_History_Overweight"],
                horizontal=True,
                label_visibility="collapsed",
                format_func=yes_no_label,
            )

            st.markdown(
                "**Frequently eats high-caloric food?**"
            )

            favc = st.radio(
                "Frequently eats high-caloric food?",
                cat_meta["Frequent_High_Caloric_Food"],
                horizontal=True,
                label_visibility="collapsed",
                format_func=yes_no_label,
            )

        with c5:

            fcvc = st.slider(
                "Vegetable consumption frequency (1 = never, 3 = always)",
                1.0,
                3.0,
                round(
                    num_meta["Vegetable_Consumption_Freq"]["mean"],
                    1,
                ),
                0.1,
            )

            ncp = st.slider(
                "Number of main meals per day",
                1.0,
                4.0,
                round(
                    num_meta["Main_Meals_Per_Day"]["mean"],
                    1,
                ),
                0.5,
            )

        with c6:

            st.markdown(
                "**Eats food between meals?**"
            )

            caec = st.select_slider(
                "Eats food between meals?",
                options=FREQUENCY_ORDER,
                value="Sometimes",
                label_visibility="collapsed",
                format_func=freq_label,
            )

            st.markdown(
                "**Alcohol consumption**"
            )

            calc = st.select_slider(
                "Alcohol consumption",
                options=FREQUENCY_ORDER,
                value="Sometimes",
                label_visibility="collapsed",
                format_func=freq_label,
            )

        # ----------------------------------------------------
        # LIFESTYLE
        # ----------------------------------------------------

        st.subheader("Lifestyle & Physical Condition")

        c7, c8, c9 = st.columns(3)

        with c7:

            st.markdown("**Smokes?**")

            smoke = st.radio(
                "Smokes?",
                cat_meta["Smokes"],
                horizontal=True,
                label_visibility="collapsed",
                format_func=yes_no_label,
            )

            st.markdown(
                "**Monitors calorie intake?**"
            )

            scc = st.radio(
                "Monitors calorie intake?",
                cat_meta["Calorie_Monitoring"],
                horizontal=True,
                label_visibility="collapsed",
                format_func=yes_no_label,
            )

        with c8:

            ch2o = st.slider(
                "Daily water intake (1 = <1L, 3 = >2L)",
                1.0,
                3.0,
                round(
                    num_meta["Daily_Water_Intake"]["mean"],
                    1,
                ),
                0.1,
            )

            faf = st.slider(
                "Physical activity frequency (0 = none, 3 = frequent)",
                0.0,
                3.0,
                round(
                    num_meta["Physical_Activity_Freq"]["mean"],
                    1,
                ),
                0.1,
            )

        with c9:

            tue = st.slider(
                "Technology usage time (0 = low, 2 = high)",
                0.0,
                2.0,
                round(
                    num_meta["Technology_Usage_Time"]["mean"],
                    1,
                ),
                0.1,
            )

            st.markdown(
                "**Usual transportation mode**"
            )

            mtrans = st.pills(
                "Usual transportation mode",
                cat_meta["Transportation_Mode"],
                default=cat_meta["Transportation_Mode"][0],
                label_visibility="collapsed",
                format_func=lambda x:
                    x.replace("_", " "),
            )

            if mtrans is None:
                mtrans = cat_meta["Transportation_Mode"][0]

        submitted = st.form_submit_button(
            "🔮 Predict Obesity Level",
            type="primary",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # PREDICTION RESULT
    # --------------------------------------------------------

    if submitted:

        input_row = pd.DataFrame(
            [
                {
                    "Gender": gender,
                    "Age": age,
                    "Height": height,
                    "Weight": weight,
                    "Family_History_Overweight": family_history,
                    "Frequent_High_Caloric_Food": favc,
                    "Vegetable_Consumption_Freq": fcvc,
                    "Main_Meals_Per_Day": ncp,
                    "Food_Between_Meals": caec,
                    "Smokes": smoke,
                    "Daily_Water_Intake": ch2o,
                    "Calorie_Monitoring": scc,
                    "Physical_Activity_Freq": faf,
                    "Technology_Usage_Time": tue,
                    "Alcohol_Consumption": calc,
                    "Transportation_Mode": mtrans,
                }
            ]
        )

        pipeline = models[chosen_model_name]

        pred_encoded = pipeline.predict(
            input_row
        )[0]

        pred_label = label_encoder.inverse_transform(
            [pred_encoded]
        )[0]

        bmi_value = weight / (height ** 2)

        st.divider()

        result_col, chart_col = st.columns(
            [1, 1.5]
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        with result_col:

            st.subheader("Prediction Result")

            st.metric(
                "Predicted Obesity Level",
                pred_label.replace("_", " "),
            )

            st.metric(
                "Computed BMI",
                f"{bmi_value:.1f} kg/m²",
            )

            st.caption(
                f"Model used: **{chosen_model_name}**"
            )

            st.markdown(
                "**Recommendation:**"
            )

            st.write(
                RECOMMENDATIONS.get(
                    pred_label,
                    "Consult a healthcare professional for guidance.",
                )
            )

        # ----------------------------------------------------
        # PROBABILITY CHART
        # ----------------------------------------------------

        with chart_col:

            if hasattr(
                pipeline,
                "predict_proba",
            ):

                proba = pipeline.predict_proba(
                    input_row
                )[0]

                proba_df = pd.DataFrame(
                    {
                        "Obesity_Level": label_encoder.classes_,
                        "Probability": proba,
                    }
                )

                proba_df = (
                    proba_df
                    .set_index("Obesity_Level")
                    .reindex(obesity_order)
                    .reset_index()
                )

                proba_df["Display_Label"] = (
                    proba_df["Obesity_Level"]
                    .str.replace("_", " ")
                )

                proba_df["Prediction"] = np.where(
                    proba_df["Obesity_Level"] == pred_label,
                    "Predicted class",
                    "Other class",
                )

                fig = px.bar(
                    proba_df,
                    x="Probability",
                    y="Display_Label",
                    orientation="h",
                    color="Prediction",
                    color_discrete_map={
                        "Predicted class": "#C44E52",
                        "Other class": "#4C72B0",
                    },
                    text="Probability",
                    title=(
                        f"Class Probabilities — "
                        f"{chosen_model_name}"
                    ),
                    category_orders={
                        "Display_Label": [
                            x.replace("_", " ")
                            for x in obesity_order
                        ]
                    },
                )

                fig.update_traces(
                    texttemplate="%{text:.1%}",
                    textposition="outside",
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Probability: %{x:.2%}<br>"
                        "%{fullData.name}"
                        "<extra></extra>"
                    ),
                )

                fig.update_xaxes(
                    range=[0, 1],
                    tickformat=".0%",
                    title="Predicted Probability",
                )

                fig.update_yaxes(
                    title="",
                )

                style_chart(
                    fig,
                    height=500,
                )

                display_chart(fig)

            else:

                st.info(
                    "This model does not expose class probabilities."
                )


# ============================================================
# TAB 3 — DATA EXPLORATION
# ============================================================

with tab_explore:

    st.header("📊 Data Exploration")

    df = load_cleaned_data()

    st.caption(
        f"Cleaned dataset: {df.shape[0]:,} rows × "
        f"{df.shape[1]} columns"
    )

    # --------------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------------

    with st.expander(
        "Preview raw table & summary statistics",
        expanded=False,
    ):

        st.dataframe(
            df.head(20),
            width="stretch",
        )

        st.write(
            "Numeric summary:"
        )

        st.dataframe(
            df.describe().T,
            width="stretch",
        )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    st.subheader("Filters")

    fc1, fc2, fc3 = st.columns(3)

    with fc1:

        gender_filter = st.multiselect(
            "Gender",
            sorted(
                df["Gender"].unique()
            ),
            default=sorted(
                df["Gender"].unique()
            ),
        )

    with fc2:

        level_filter = st.multiselect(
            "Obesity Level",
            obesity_order,
            default=obesity_order,
        )

    with fc3:

        age_range = st.slider(
            "Age range",
            int(df["Age"].min()),
            int(df["Age"].max()),
            (
                int(df["Age"].min()),
                int(df["Age"].max()),
            ),
        )

    filtered = df[
        df["Gender"].isin(gender_filter)
        & df["Obesity_Level"].isin(level_filter)
        & df["Age"].between(*age_range)
    ]

    st.caption(
        f"Showing {len(filtered):,} of "
        f"{len(df):,} records after filtering."
    )

    if filtered.empty:

        st.warning(
            "No records match the selected filters."
        )

        st.stop()

    # ========================================================
    # CHART 1 + CHART 2
    # ========================================================

    chart1, chart2 = st.columns(2)

    # --------------------------------------------------------
    # PIE CHART
    # --------------------------------------------------------

    with chart1:

        st.markdown(
            "**Obesity Level Distribution**"
        )

        counts = (
            filtered["Obesity_Level"]
            .value_counts()
            .reindex(obesity_order)
            .dropna()
            .reset_index()
        )

        counts.columns = [
            "Obesity_Level",
            "Count",
        ]

        counts["Display_Label"] = (
            counts["Obesity_Level"]
            .str.replace("_", " ")
        )

        fig = px.pie(
            counts,
            names="Display_Label",
            values="Count",
            title="Obesity Level Distribution",
            color="Obesity_Level",
            color_discrete_map=OBESITY_COLORS,
            hole=0.35,
        )

        fig.update_traces(
            textinfo="percent",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Count: %{value}<br>"
                "Percentage: %{percent}"
                "<extra></extra>"
            ),
        )

        style_chart(
            fig,
            height=520,
        )

        display_chart(fig)

    # --------------------------------------------------------
    # HISTOGRAM
    # --------------------------------------------------------

    with chart2:

        st.markdown(
            "**Numeric Distribution**"
        )

        numeric_cols = (
            filtered
            .select_dtypes(include=np.number)
            .columns
            .tolist()
        )

        default_hist_index = (
            numeric_cols.index("BMI")
            if "BMI" in numeric_cols
            else 0
        )

        hist_col = st.selectbox(
            "Choose a numeric column",
            numeric_cols,
            index=default_hist_index,
        )

        fig = px.histogram(
            filtered,
            x=hist_col,
            nbins=20,
            marginal="box",
            title=f"Distribution of {hist_col}",
        )

        fig.update_traces(
            hovertemplate=(
                f"<b>{hist_col}</b>: "
                "%{x}<br>"
                "Count: %{y}"
                "<extra></extra>"
            ),
        )

        style_chart(
            fig,
            height=520,
        )

        display_chart(fig)

    # ========================================================
    # CHART 3 + CHART 4
    # ========================================================

    chart3, chart4 = st.columns(2)

    # --------------------------------------------------------
    # SCATTERPLOT
    # --------------------------------------------------------

    with chart3:

        st.markdown(
            "**Interactive Scatterplot**"
        )

        num_options = (
            filtered
            .select_dtypes(include=np.number)
            .columns
            .tolist()
        )

        default_x = (
            num_options.index("Height")
            if "Height" in num_options
            else 0
        )

        default_y = (
            num_options.index("Weight")
            if "Weight" in num_options
            else min(1, len(num_options) - 1)
        )

        sx = st.selectbox(
            "X-axis",
            num_options,
            index=default_x,
            key="sx",
        )

        sy = st.selectbox(
            "Y-axis",
            num_options,
            index=default_y,
            key="sy",
        )

        hover_columns = [
            "Gender",
            "Age",
            "Height",
            "Weight",
            "BMI",
            "Obesity_Level",
        ]

        hover_columns = [
            c for c in hover_columns
            if c in filtered.columns
        ]

        fig = px.scatter(
            filtered,
            x=sx,
            y=sy,
            color="Obesity_Level",
            color_discrete_map=OBESITY_COLORS,
            category_orders={
                "Obesity_Level": obesity_order
            },
            hover_data=hover_columns,
            title=f"{sx} vs {sy}",
            opacity=0.75,
        )

        fig.update_traces(
            marker=dict(
                size=8,
            ),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                f"{sx}: %{{x}}<br>"
                f"{sy}: %{{y}}"
                "<extra></extra>"
            ),
        )

        style_chart(
            fig,
            height=550,
        )

        display_chart(fig)

    # --------------------------------------------------------
    # BOXPLOT
    # --------------------------------------------------------

    with chart4:

        st.markdown(
            "**Boxplot by Obesity Level**"
        )

        box_col = st.selectbox(
            "Numeric column",
            num_options,
            index=(
                num_options.index("Weight")
                if "Weight" in num_options
                else 0
            ),
            key="box",
        )

        fig = px.box(
            filtered,
            x="Obesity_Level",
            y=box_col,
            color="Obesity_Level",
            color_discrete_map=OBESITY_COLORS,
            category_orders={
                "Obesity_Level": obesity_order
            },
            points="outliers",
            title=f"{box_col} by Obesity Level",
        )

        fig.update_traces(
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"{box_col}: %{{y}}"
                "<extra></extra>"
            ),
        )

        fig.update_xaxes(
            tickangle=-35,
        )

        style_chart(
            fig,
            height=550,
        )

        display_chart(fig)

    # ========================================================
    # CORRELATION HEATMAP
    # ========================================================

    st.subheader("Correlation Heatmap")

    numeric_data = (
        filtered
        .select_dtypes(include=np.number)
    )

    if numeric_data.shape[1] >= 2:

        corr = numeric_data.corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Feature Correlation Matrix",
        )

        fig.update_traces(
            hovertemplate=(
                "<b>%{y}</b> vs <b>%{x}</b><br>"
                "Correlation: %{z:.2f}"
                "<extra></extra>"
            ),
        )

        fig.update_layout(
            coloraxis_colorbar=dict(
                title="Correlation"
            )
        )

        style_chart(
            fig,
            height=650,
        )

        display_chart(fig)

    else:

        st.info(
            "At least two numeric variables are required "
            "to generate a correlation heatmap."
        )

    # ========================================================
    # OLAP-STYLE PIVOT EXPLORER
    # ========================================================

    st.subheader(
        "OLAP-style Pivot Explorer "
        "(Roll-up / Drill-down)"
    )

    st.caption(
        "Build your own multidimensional summary table, similar to an OLAP cube: "
        "choose row and column dimensions, a numeric measure, and an aggregation function."
    )

    categorical_options = (
        filtered
        .select_dtypes(exclude=np.number)
        .columns
        .tolist()
    )

    p1, p2, p3, p4 = st.columns(4)

    with p1:

        row_dim = st.selectbox(
            "Row dimension",
            categorical_options,
            index=(
                categorical_options.index("Gender")
                if "Gender" in categorical_options
                else 0
            ),
        )

    with p2:

        col_dim = st.selectbox(
            "Column dimension",
            categorical_options,
            index=(
                categorical_options.index("Obesity_Level")
                if "Obesity_Level" in categorical_options
                else 0
            ),
        )

    with p3:

        measure = st.selectbox(
            "Measure",
            num_options,
            index=(
                num_options.index("BMI")
                if "BMI" in num_options
                else 0
            ),
        )

    with p4:

        agg_func = st.selectbox(
            "Aggregation",
            [
                "mean",
                "count",
                "sum",
                "median",
            ],
        )

    if row_dim != col_dim:

        pivot = pd.pivot_table(
            filtered,
            index=row_dim,
            columns=col_dim,
            values=measure,
            aggfunc=agg_func,
            margins=True,
            margins_name="All (Roll-up)",
        ).round(2)

        st.markdown(
            "**Pivot Summary Table**"
        )

        st.dataframe(
            pivot,
            width="stretch",
        )

        # ----------------------------------------------------
        # PIVOT VISUALISATION
        # ----------------------------------------------------

        st.markdown(
            "**Interactive Pivot Visualisation**"
        )

        pivot_chart = (
            pivot
            .drop(
                index="All (Roll-up)",
                errors="ignore",
            )
            .copy()
        )

        if "All (Roll-up)" in pivot_chart.columns:

            pivot_chart = pivot_chart.drop(
                columns="All (Roll-up)"
            )

        pivot_long = (
            pivot_chart
            .reset_index()
            .melt(
                id_vars=row_dim,
                var_name=col_dim,
                value_name=measure,
            )
        )

        fig = px.bar(
            pivot_long,
            x=row_dim,
            y=measure,
            color=col_dim,
            barmode="group",
            title=(
                f"{agg_func.title()} of {measure} "
                f"by {row_dim} and {col_dim}"
            ),
        )

        fig.update_traces(
            hovertemplate=(
                "<b>%{x}</b><br>"
                "%{fullData.name}<br>"
                f"{measure}: %{{y:.2f}}"
                "<extra></extra>"
            ),
        )

        style_chart(
            fig,
            height=520,
        )

        display_chart(fig)

    else:

        st.warning(
            "Choose two different dimensions for rows and columns."
        )


# ============================================================
# TAB 4 — MODEL PERFORMANCE
# ============================================================

with tab_performance:

    st.header("📈 Model Performance")

    comparison_df = load_comparison_table()

    # ========================================================
    # CONSOLIDATED TABLE
    # ========================================================

    st.subheader(
        "Consolidated Comparison Table"
    )

    st.caption(
        "`Train_Accuracy_InSample` is the model scored on the exact rows it was fit on. "
        "It is expected to be very high for flexible models and is not a reliable "
        "overfitting signal by itself. `Train_Accuracy_CV` is cross-validated on held-out "
        "training folds and is the more meaningful comparison."
    )

    st.dataframe(
        comparison_df.style.format(
            precision=4
        ),
        width="stretch",
    )

    # ========================================================
    # OUTER COMPARISON
    # ========================================================

    st.subheader(
        "Outer Comparison — Models Against Each Other "
        "(Test Set)"
    )

    outer_metrics = [
        "Test_Accuracy",
        "Precision_Weighted",
        "Recall_Weighted",
        "F1_Weighted",
    ]

    outer_long = (
        comparison_df[outer_metrics]
        .reset_index()
        .melt(
            id_vars="index",
            var_name="Metric",
            value_name="Score",
        )
        .rename(
            columns={
                "index": "Model"
            }
        )
    )

    metric_display_names = {
        "Test_Accuracy": "Accuracy",
        "Precision_Weighted": "Precision",
        "Recall_Weighted": "Recall",
        "F1_Weighted": "F1",
    }

    outer_long["Metric"] = (
        outer_long["Metric"]
        .map(metric_display_names)
        .fillna(outer_long["Metric"])
    )

    fig = px.bar(
        outer_long,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text="Score",
        title="Model Performance — Test Set",
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "%{fullData.name}: %{y:.3f}"
            "<extra></extra>"
        ),
    )

    fig.update_yaxes(
        range=[0, 1.05],
        tickformat=".0%",
        title="Score",
    )

    fig.update_xaxes(
        tickangle=-20,
    )

    style_chart(
        fig,
        height=550,
    )

    display_chart(fig)

    # ========================================================
    # INNER COMPARISON
    # ========================================================

    st.subheader(
        "Inner Comparison — Train vs Test Accuracy "
        "(Overfitting Check)"
    )

    inner_tab1, inner_tab2 = st.tabs(
        [
            "Cross-validated (trust this)",
            "In-sample (for transparency)",
        ]
    )

    # --------------------------------------------------------
    # CROSS-VALIDATED
    # --------------------------------------------------------

    with inner_tab1:

        train_test_long = (
            comparison_df[
                [
                    "Train_Accuracy_CV",
                    "Test_Accuracy",
                ]
            ]
            .reset_index()
            .melt(
                id_vars="index",
                var_name="Dataset",
                value_name="Accuracy",
            )
            .rename(
                columns={
                    "index": "Model"
                }
            )
        )

        train_test_long["Dataset"] = (
            train_test_long["Dataset"]
            .replace(
                {
                    "Train_Accuracy_CV":
                        "Training — Cross-Validated",
                    "Test_Accuracy":
                        "Testing — Held-Out",
                }
            )
        )

        fig = px.bar(
            train_test_long,
            x="Model",
            y="Accuracy",
            color="Dataset",
            barmode="group",
            text="Accuracy",
            title="Cross-Validated Train vs Test Accuracy",
        )

        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "%{fullData.name}: %{y:.3f}"
                "<extra></extra>"
            ),
        )

        fig.update_yaxes(
            range=[0, 1.05],
            tickformat=".0%",
        )

        fig.update_xaxes(
            tickangle=-20,
        )

        style_chart(
            fig,
            height=550,
        )

        display_chart(fig)

        st.caption(
            "Each fold's model is scored on held-out rows of the training data, "
            "so this number cannot trivially reach 1.0 through memorisation. "
            "This is the meaningful train-vs-test overfitting comparison."
        )

    # --------------------------------------------------------
    # IN-SAMPLE
    # --------------------------------------------------------

    with inner_tab2:

        insample_long = (
            comparison_df[
                [
                    "Train_Accuracy_InSample",
                    "Test_Accuracy",
                ]
            ]
            .reset_index()
            .melt(
                id_vars="index",
                var_name="Dataset",
                value_name="Accuracy",
            )
            .rename(
                columns={
                    "index": "Model"
                }
            )
        )

        insample_long["Dataset"] = (
            insample_long["Dataset"]
            .replace(
                {
                    "Train_Accuracy_InSample":
                        "Training — In-Sample",
                    "Test_Accuracy":
                        "Testing — Held-Out",
                }
            )
        )

        fig = px.bar(
            insample_long,
            x="Model",
            y="Accuracy",
            color="Dataset",
            barmode="group",
            text="Accuracy",
            title="In-Sample Train vs Test Accuracy",
        )

        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "%{fullData.name}: %{y:.3f}"
                "<extra></extra>"
            ),
        )

        fig.update_yaxes(
            range=[0, 1.05],
            tickformat=".0%",
        )

        fig.update_xaxes(
            tickangle=-20,
        )

        style_chart(
            fig,
            height=550,
        )

        display_chart(fig)

        st.caption(
            "In-sample accuracy means fitting and predicting on the same training rows. "
            "It mainly measures memorisation rather than generalisation, so very high "
            "values are expected for flexible models."
        )

    # ========================================================
    # WEIGHTED VS MACRO
    # ========================================================

    st.subheader(
        "Weighted vs Macro-averaged Metrics"
    )

    st.caption(
        "Weighted Recall is mathematically identical to Accuracy for a single-label "
        "multiclass problem. Macro metrics give every class equal importance and therefore "
        "show more clearly when a model performs poorly on smaller or difficult classes."
    )

    macro_metric_tabs = st.tabs(
        [
            "Precision",
            "Recall",
            "F1",
        ]
    )

    for tab, metric in zip(
        macro_metric_tabs,
        [
            "Precision",
            "Recall",
            "F1",
        ],
    ):

        with tab:

            metric_columns = [
                f"{metric}_Weighted",
                f"{metric}_Macro",
            ]

            metric_df = (
                comparison_df[
                    metric_columns
                ]
                .reset_index()
                .melt(
                    id_vars="index",
                    var_name="Average",
                    value_name="Score",
                )
                .rename(
                    columns={
                        "index": "Model"
                    }
                )
            )

            metric_df["Average"] = (
                metric_df["Average"]
                .str.replace(
                    f"{metric}_",
                    "",
                    regex=False,
                )
                .str.title()
            )

            fig = px.bar(
                metric_df,
                x="Model",
                y="Score",
                color="Average",
                barmode="group",
                text="Score",
                title=(
                    f"{metric}: "
                    "Weighted vs Macro"
                ),
            )

            fig.update_traces(
                texttemplate="%{text:.3f}",
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "%{fullData.name}: %{y:.3f}"
                    "<extra></extra>"
                ),
            )

            fig.update_yaxes(
                range=[0, 1.05],
                tickformat=".0%",
            )

            fig.update_xaxes(
                tickangle=-20,
            )

            style_chart(
                fig,
                height=480,
            )

            display_chart(fig)

    # ========================================================
    # LIVE EVALUATION
    # ========================================================

    st.divider()

    st.subheader(
        "Live Evaluation for a Selected Model"
    )

    st.caption(
        "Recomputed live on the same held-out test split used in the notebook "
        "(identical random_state, test_size, and stratification)."
    )

    eval_model_options = list(
        models.keys()
    )

    eval_model_name = st.selectbox(
        "Choose a model to inspect",
        eval_model_options,
        index=(
            eval_model_options.index(
                best_model_name
            )
            if best_model_name in eval_model_options
            else 0
        ),
        key="eval_model",
    )

    X_train, X_test, y_train, y_test = (
        rebuild_test_split(metadata)
    )

    eval_pipeline = models[
        eval_model_name
    ]

    y_pred = eval_pipeline.predict(
        X_test
    )

    # ========================================================
    # WEIGHTED METRICS
    # ========================================================

    st.markdown(
        "**Weighted-average metrics**"
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Test Accuracy",
        f"{accuracy_score(y_test, y_pred):.1%}",
    )

    m2.metric(
        "Precision",
        f"{precision_score(
            y_test,
            y_pred,
            average='weighted',
            zero_division=0
        ):.1%}",
    )

    m3.metric(
        "Recall",
        f"{recall_score(
            y_test,
            y_pred,
            average='weighted',
            zero_division=0
        ):.1%}",
    )

    m4.metric(
        "F1",
        f"{f1_score(
            y_test,
            y_pred,
            average='weighted',
            zero_division=0
        ):.1%}",
    )

    # ========================================================
    # MACRO METRICS
    # ========================================================

    st.markdown(
        "**Macro-average metrics** "
        "(every class counted equally)"
    )

    m5, m6, m7, m8 = st.columns(4)

    m5.metric(
        "Model",
        eval_model_name,
    )

    m6.metric(
        "Precision",
        f"{precision_score(
            y_test,
            y_pred,
            average='macro',
            zero_division=0
        ):.1%}",
    )

    m7.metric(
        "Recall",
        f"{recall_score(
            y_test,
            y_pred,
            average='macro',
            zero_division=0
        ):.1%}",
    )

    m8.metric(
        "F1",
        f"{f1_score(
            y_test,
            y_pred,
            average='macro',
            zero_division=0
        ):.1%}",
    )

    # ========================================================
    # CONFUSION MATRIX + ROC
    # ========================================================

    cm_col, roc_col = st.columns(2)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    with cm_col:

        st.markdown(
            "**Interactive Confusion Matrix**"
        )

        cm = confusion_matrix(
            y_test,
            y_pred,
        )

        class_names = (
            label_encoder.classes_
        )

        display_class_names = [
            x.replace("_", " ")
            for x in class_names
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=display_class_names,
                y=display_class_names,
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                hovertemplate=(
                    "<b>Actual:</b> %{y}<br>"
                    "<b>Predicted:</b> %{x}<br>"
                    "<b>Count:</b> %{z}"
                    "<extra></extra>"
                ),
                colorbar=dict(
                    title="Count"
                ),
            )
        )

        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            template=CHART_TEMPLATE,
            height=650,
            margin=dict(
                l=100,
                r=30,
                t=75,
                b=140,
            ),
        )

        fig.update_xaxes(
            tickangle=-40,
        )

        display_chart(fig)

    # --------------------------------------------------------
    # ROC CURVES
    # --------------------------------------------------------

    with roc_col:

        st.markdown(
            "**Interactive Per-class ROC Curves**"
        )

        if hasattr(
            eval_pipeline,
            "predict_proba",
        ):

            y_proba = (
                eval_pipeline
                .predict_proba(X_test)
            )

            y_test_bin = label_binarize(
                y_test,
                classes=list(
                    range(
                        len(
                            label_encoder.classes_
                        )
                    )
                ),
            )

            fig = go.Figure()

            for i, class_name in enumerate(
                label_encoder.classes_
            ):

                fpr, tpr, _ = roc_curve(
                    y_test_bin[:, i],
                    y_proba[:, i],
                )

                roc_auc_i = auc(
                    fpr,
                    tpr,
                )

                display_name = (
                    class_name
                    .replace("_", " ")
                )

                fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        name=(
                            f"{display_name} "
                            f"(AUC={roc_auc_i:.2f})"
                        ),
                        hovertemplate=(
                            f"<b>{display_name}</b><br>"
                            "False Positive Rate: %{x:.3f}<br>"
                            "True Positive Rate: %{y:.3f}<br>"
                            f"AUC: {roc_auc_i:.3f}"
                            "<extra></extra>"
                        ),
                    )
                )

            # Random classifier reference line

            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Random Classifier",
                    line=dict(
                        dash="dot"
                    ),
                    hoverinfo="skip",
                )
            )

            fig.update_layout(
                title="Per-Class ROC Curves",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                xaxis=dict(
                    range=[0, 1]
                ),
                yaxis=dict(
                    range=[0, 1]
                ),
                template=CHART_TEMPLATE,
                height=650,
                hovermode="x unified",
                margin=dict(
                    l=60,
                    r=30,
                    t=75,
                    b=80,
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                ),
            )

            display_chart(fig)

        else:

            st.info(
                "This model does not expose class probabilities "
                "and therefore cannot generate ROC curves."
            )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    with st.expander(
        "📋 Full classification report"
    ):

        report = classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
            output_dict=True,
        )

        report_df = (
            pd.DataFrame(report)
            .T
            .round(3)
        )

        st.dataframe(
            report_df,
            width="stretch",
        )

    # ========================================================
    # RANDOM FOREST FEATURE IMPORTANCE
    # ========================================================

    if eval_model_name == "Random Forest":

        st.divider()

        st.subheader(
            "🌳 Random Forest — Feature Importance"
        )

        try:

            feature_names = (
                eval_pipeline
                .named_steps[
                    "preprocessor"
                ]
                .get_feature_names_out()
            )

            importances = (
                eval_pipeline
                .named_steps[
                    "classifier"
                ]
                .feature_importances_
            )

            importance_df = (
                pd.DataFrame(
                    {
                        "Feature": feature_names,
                        "Importance": importances,
                    }
                )
                .sort_values(
                    "Importance",
                    ascending=False,
                )
                .head(15)
                .sort_values(
                    "Importance"
                )
            )

            importance_df["Display_Feature"] = (
                importance_df["Feature"]
                .str.replace(
                    "num__",
                    "",
                    regex=False,
                )
                .str.replace(
                    "cat__",
                    "",
                    regex=False,
                )
                .str.replace(
                    "_",
                    " ",
                    regex=False,
                )
            )

            fig = px.bar(
                importance_df,
                x="Importance",
                y="Display_Feature",
                orientation="h",
                text="Importance",
                title=(
                    "Top 15 Feature Importances"
                ),
            )

            fig.update_traces(
                texttemplate="%{text:.3f}",
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Importance: %{x:.4f}"
                    "<extra></extra>"
                ),
            )

            fig.update_xaxes(
                title="Feature Importance"
            )

            fig.update_yaxes(
                title=""
            )

            style_chart(
                fig,
                height=650,
            )

            display_chart(fig)

        except Exception as e:

            st.warning(
                "Feature importance could not be displayed "
                f"for this saved Random Forest pipeline: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "BMDS2003 Data Science — Group Project | "
    "CRISP-DM Prototype | Built with Streamlit + Plotly"
)
```
