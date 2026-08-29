"""
BMDS2003 Data Science — Obesity Level Prediction App
======================================================
A Streamlit deployment prototype for the group project
"Estimation of Obesity Levels Based on Eating Habits and Physical Condition".

Run locally with:
    streamlit run app.py
"""

import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split


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
# PATHS
# ============================================================

MODELS_DIR = Path(__file__).parent / "models"


# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILES = {
    "Decision Tree (Baseline)": "decision_tree_pipeline.pkl",
    "Random Forest": "random_forest_pipeline.pkl",
    "SVM": "svm_pipeline.pkl",
    "KNN": "knn_pipeline.pkl",
}


# ============================================================
# RECOMMENDATIONS
# ============================================================

RECOMMENDATIONS = {

    "Insufficient_Weight": (
        "Your inputs suggest an underweight profile. Consider a structured "
        "nutrition plan that gradually increases calorie-dense, nutritious "
        "foods, and consult a healthcare professional to rule out underlying causes."
    ),

    "Normal_Weight": (
        "Your inputs suggest a healthy weight range. Keep up balanced meals, "
        "regular physical activity, and adequate water intake to maintain this."
    ),

    "Overweight_Level_I": (
        "Your inputs suggest early-stage overweight. Small, sustainable changes "
        "such as eating more vegetables, reducing high-calorie snacks, and "
        "increasing physical activity can help reduce the risk of progressing further."
    ),

    "Overweight_Level_II": (
        "Your inputs suggest overweight. A structured plan combining dietary "
        "adjustments, increased physical activity, and reduced sedentary technology "
        "time is recommended, ideally with professional guidance."
    ),

    "Obesity_Type_I": (
        "Your inputs suggest Class I obesity. We recommend consulting a healthcare "
        "provider or dietitian to design a supervised weight-management plan, "
        "alongside gradual increases in physical activity."
    ),

    "Obesity_Type_II": (
        "Your inputs suggest Class II obesity. Professional medical guidance is "
        "strongly recommended to design a safe, supervised intervention plan."
    ),

    "Obesity_Type_III": (
        "Your inputs suggest Class III obesity. Please consult a healthcare "
        "professional to discuss a comprehensive, medically supervised management plan."
    ),
}


# ============================================================
# OBESITY ORDER
# ============================================================

OBESITY_COLORS = {
    "Insufficient_Weight": "#3498DB",
    "Normal_Weight": "#2ECC71",
    "Overweight_Level_I": "#F1C40F",
    "Overweight_Level_II": "#E67E22",
    "Obesity_Type_I": "#E74C3C",
    "Obesity_Type_II": "#9B59B6",
    "Obesity_Type_III": "#8E44AD",
}

ORDINAL_ORDER = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]


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

    label_encoder = joblib.load(
        MODELS_DIR / "label_encoder.pkl"
    )

    return models, label_encoder


@st.cache_resource(show_spinner=False)
def load_metadata():

    with open(
        MODELS_DIR / "feature_metadata.json",
        encoding="utf-8"
    ) as f:

        return json.load(f)


@st.cache_data(show_spinner="Loading dataset...")
def load_cleaned_data():

    return pd.read_csv(
        MODELS_DIR / "obesity_cleaned.csv"
    )


@st.cache_data(show_spinner=False)
def load_comparison_table():

    return pd.read_csv(
        MODELS_DIR / "model_comparison.csv",
        index_col=0
    )


@st.cache_data(
    show_spinner="Rebuilding the held-out test split for evaluation..."
)
def rebuild_test_split(_metadata):

    """
    Recreate the exact same train/test split used in the notebook.
    """

    df = load_cleaned_data()

    exclude_cols = [
        "Obesity_Level",
        "BMI",
        "Age_Group"
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

    y_encoded = y.map(
        class_to_int
    ).values

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y_encoded,

        test_size=0.20,

        random_state=42,

        stratify=y_encoded
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# LOAD EVERYTHING
# ============================================================

models, label_encoder = load_models()

metadata = load_metadata()


# ============================================================
# SESSION STATE — PREDICTION HISTORY
# ============================================================

if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []


if "prediction_reset_counter" not in st.session_state:

    st.session_state.prediction_reset_counter = 0


# ============================================================
# GENERAL SETTINGS
# ============================================================

obesity_order = metadata.get(
    "obesity_order",
    ORDINAL_ORDER
)

best_model_name = metadata.get(
    "best_model",
    "Random Forest"
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "🍎 Obesity Level Prediction System"
)

st.caption(
    "BMDS2003 Data Science Group Project — "
    "Estimation of Obesity Levels Based on Eating Habits "
    "and Physical Condition (CRISP-DM)"
)


# ============================================================
# TABS
# ============================================================

tab_about, tab_predict, tab_explore, tab_history, tab_performance = st.tabs(
    [
        "🏠 About",
        "🔮 Prediction",
        "📊 Data Exploration",
        "🕒 History",
        "📈 Model Performance"
    ]
)


# ============================================================
# TAB 1 — ABOUT
# ============================================================

with tab_about:

    st.header(
        "About This Project"
    )

    col1, col2 = st.columns(
        [2, 1]
    )


    # ========================================================
    # LEFT COLUMN
    # ========================================================

    with col1:

        st.subheader(
            "Business Understanding"
        )

        st.markdown(
            """
Obesity is a growing public-health concern linked to diabetes,
cardiovascular disease, and reduced quality of life.

This project explores whether **everyday lifestyle and eating habits**
can predict a person's obesity category.

The system can support:

- **Self-assessment tools** that flag potential obesity risk.
- **Public-health screening** at scale.
- **Targeted lifestyle recommendations** based on eating habits,
  activity, and physical-condition variables.

**Objective:** classify an individual into one of seven obesity levels
using lifestyle, dietary, and physical-condition attributes.
"""
        )


        # ====================================================
        # DATASET
        # ====================================================

        st.subheader(
            "Dataset"
        )

        st.markdown(
            """
- **Source:** UCI Machine Learning Repository — *Estimation of Obesity
  Levels Based on Eating Habits and Physical Condition*.
- **Records:** 2,111 respondents.
- **Variables:** 16 predictive features plus the target.
- **Target:** `Obesity_Level`
- **Classes:** 7 obesity categories ranging from
  `Insufficient_Weight` to `Obesity_Type_III`.
"""
        )


        # ====================================================
        # CRISP-DM
        # ====================================================

        st.subheader(
            "CRISP-DM Workflow"
        )

        st.markdown(
            """
1. **Business Understanding** — define the prediction problem.
2. **Data Understanding** — inspect the dataset structure,
   distributions, and relationships.
3. **Data Preparation** — clean the data, handle outliers,
   engineer BMI for analysis, and prepare model features.
4. **Modelling** — train four classifiers:
   Decision Tree, Random Forest, SVM, and KNN.
5. **Evaluation** — compare accuracy, precision, recall, F1,
   AUC, confusion matrices, ROC curves, and overfitting.
6. **Deployment** — provide this interactive Streamlit application.
"""
        )


    # ========================================================
    # RIGHT COLUMN
    # ========================================================

    with col2:

        st.subheader(
            "Team"
        )

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


        # ====================================================
        # BEST MODEL
        # ====================================================

        st.subheader(
            "Best Model"
        )

        st.success(
            f"**{best_model_name}** achieved the highest test-set accuracy."
        )

        comparison_preview = load_comparison_table()

        if best_model_name in comparison_preview.index:

            st.metric(
                "Best Test Accuracy",
                f"{comparison_preview.loc[best_model_name, 'Test_Accuracy']:.1%}"
            )


        st.info(
            "Use the **Prediction** tab to generate a prediction, "
            "the **Data Exploration** tab to explore the dataset, "
            "the **History** tab to review previous predictions, "
            "and the **Model Performance** tab to inspect model evaluation."
        )


# ============================================================
# TAB 2 — PREDICTION
# ============================================================

with tab_predict:

    st.header(
        "🔮 Predict an Obesity Level"
    )

    st.caption(
        "Fill in the fields below and choose a model to generate "
        "a live prediction using the saved preprocessing and classifier pipeline."
    )


    # ========================================================
    # METADATA
    # ========================================================

    num_meta = metadata["numeric_features"]

    cat_meta = metadata["categorical_features"]


    # ========================================================
    # FREQUENCY OPTIONS
    # ========================================================

    FREQUENCY_ORDER = [
        "no",
        "Sometimes",
        "Frequently",
        "Always"
    ]


    def freq_label(x):

        return "Never" if x == "no" else x


    def yes_no_label(x):

        return (
            "✅ Yes"
            if x == "yes"
            else "❌ No"
        )


    # ========================================================
    # AVAILABLE MODELS
    # ========================================================

    available_models = [
        m
        for m in MODEL_FILES
        if m in models
    ]

    if not available_models:

        st.error(
            "No trained models were found in the models folder."
        )

        st.stop()


    default_model_index = (

        available_models.index(
            best_model_name
        )

        if best_model_name in available_models

        else 0
    )


    # ========================================================
    # RESET COUNTER
    # ========================================================

    reset_counter = (
        st.session_state.prediction_reset_counter
    )


    # ========================================================
    # MODEL SELECTION
    # ========================================================

    st.markdown(
        "**Choose a model for prediction**"
    )

    chosen_model_name = st.segmented_control(

        "Choose a model for prediction",

        available_models,

        default=available_models[
            default_model_index
        ],

        label_visibility="collapsed",

        key=f"prediction_model_{reset_counter}"
    )


    if chosen_model_name is None:

        chosen_model_name = available_models[
            default_model_index
        ]


    st.caption(
        f"Using **{chosen_model_name}**"
        +
        (
            " (best on test set)"
            if chosen_model_name == best_model_name
            else ""
        )
    )


    # ========================================================
    # PREDICTION FORM
    # ========================================================

    with st.form(
        f"prediction_form_{reset_counter}"
    ):


        # ====================================================
        # PERSONAL & PHYSICAL ATTRIBUTES
        # ====================================================

        st.subheader(
            "Personal & Physical Attributes"
        )

        c1, c2, c3 = st.columns(3)


        with c1:

            st.markdown(
                "**Gender**"
            )

            gender = st.radio(

                "Gender",

                cat_meta["Gender"],

                horizontal=True,

                label_visibility="collapsed",

                format_func=lambda x:
                    "👩 Female"
                    if x == "Female"
                    else "👨 Male",

                key=f"gender_{reset_counter}"
            )


            age = st.number_input(

                "Age (years)",

                min_value=float(
                    num_meta["Age"]["min"]
                ),

                max_value=100.0,

                value=round(
                    num_meta["Age"]["mean"],
                    1
                ),

                step=1.0,

                key=f"age_{reset_counter}"
            )


        with c2:

            height = st.number_input(

                "Height (m)",

                min_value=float(
                    num_meta["Height"]["min"]
                ),

                max_value=float(
                    num_meta["Height"]["max"]
                ) + 0.3,

                value=round(
                    num_meta["Height"]["mean"],
                    2
                ),

                step=0.01,

                format="%.2f",

                key=f"height_{reset_counter}"
            )


        with c3:

            weight = st.number_input(

                "Weight (kg)",

                min_value=float(
                    num_meta["Weight"]["min"]
                ),

                max_value=float(
                    num_meta["Weight"]["max"]
                ) + 50.0,

                value=round(
                    num_meta["Weight"]["mean"],
                    1
                ),

                step=1.0,

                key=f"weight_{reset_counter}"
            )


        # ====================================================
        # EATING HABITS
        # ====================================================

        st.subheader(
            "Eating Habits"
        )

        c4, c5, c6 = st.columns(3)


        with c4:

            st.markdown(
                "**Family history of overweight?**"
            )

            family_history = st.radio(

                "Family history of overweight?",

                cat_meta[
                    "Family_History_Overweight"
                ],

                horizontal=True,

                label_visibility="collapsed",

                format_func=yes_no_label,

                key=f"family_history_{reset_counter}"
            )


            st.markdown(
                "**Frequently eats high-caloric food?**"
            )

            favc = st.radio(

                "Frequently eats high-caloric food?",

                cat_meta[
                    "Frequent_High_Caloric_Food"
                ],

                horizontal=True,

                label_visibility="collapsed",

                format_func=yes_no_label,

                key=f"favc_{reset_counter}"
            )


        with c5:

            fcvc = st.slider(

                "Vegetable consumption frequency "
                "(1 = never, 3 = always)",

                min_value=1.0,

                max_value=3.0,

                value=round(
                    num_meta[
                        "Vegetable_Consumption_Freq"
                    ]["mean"],
                    1
                ),

                step=0.1,

                key=f"fcvc_{reset_counter}"
            )


            ncp = st.slider(

                "Number of main meals per day",

                min_value=1.0,

                max_value=4.0,

                value=round(
                    num_meta[
                        "Main_Meals_Per_Day"
                    ]["mean"],
                    1
                ),

                step=0.5,

                key=f"ncp_{reset_counter}"
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

                key=f"caec_{reset_counter}"
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

                key=f"calc_{reset_counter}"
            )


        # ====================================================
        # LIFESTYLE
        # ====================================================

        st.subheader(
            "Lifestyle & Physical Condition"
        )

        c7, c8, c9 = st.columns(3)


        with c7:

            st.markdown(
                "**Smokes?**"
            )

            smoke = st.radio(

                "Smokes?",

                cat_meta["Smokes"],

                horizontal=True,

                label_visibility="collapsed",

                format_func=yes_no_label,

                key=f"smoke_{reset_counter}"
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

                key=f"scc_{reset_counter}"
            )


        with c8:

            ch2o = st.slider(

                "Daily water intake "
                "(1 = <1L, 3 = >2L)",

                min_value=1.0,

                max_value=3.0,

                value=round(
                    num_meta[
                        "Daily_Water_Intake"
                    ]["mean"],
                    1
                ),

                step=0.1,

                key=f"ch2o_{reset_counter}"
            )


            faf = st.slider(

                "Physical activity frequency "
                "(0 = none, 3 = frequent)",

                min_value=0.0,

                max_value=3.0,

                value=round(
                    num_meta[
                        "Physical_Activity_Freq"
                    ]["mean"],
                    1
                ),

                step=0.1,

                key=f"faf_{reset_counter}"
            )


        with c9:

            tue = st.slider(

                "Technology usage time "
                "(0 = low, 2 = high)",

                min_value=0.0,

                max_value=2.0,

                value=round(
                    num_meta[
                        "Technology_Usage_Time"
                    ]["mean"],
                    1
                ),

                step=0.1,

                key=f"tue_{reset_counter}"
            )


            st.markdown(
                "**Usual transportation mode**"
            )

            mtrans = st.pills(

                "Usual transportation mode",

                cat_meta[
                    "Transportation_Mode"
                ],

                default=cat_meta[
                    "Transportation_Mode"
                ][0],

                label_visibility="collapsed",

                format_func=lambda x:
                    x.replace("_", " "),

                key=f"mtrans_{reset_counter}"
            )


            if mtrans is None:

                mtrans = cat_meta[
                    "Transportation_Mode"
                ][0]


        # ====================================================
        # BUTTONS
        # ====================================================

        st.markdown("")

        predict_col, restart_col = st.columns(2)


        with predict_col:

            submitted = st.form_submit_button(

                "🔮 Predict Obesity Level",

                type="primary",

                use_container_width=True
            )


        with restart_col:

            restart = st.form_submit_button(

                "🔄 Restart",

                type="secondary",

                use_container_width=True
            )


    # ========================================================
    # RESTART
    # ========================================================

    if restart:

        st.session_state.prediction_reset_counter += 1

        st.rerun()


    # ========================================================
    # PREDICTION
    # ========================================================

    if submitted:

        # ====================================================
        # INPUT DATA
        # ====================================================

        input_row = pd.DataFrame([{

            "Gender": gender,

            "Age": age,

            "Height": height,

            "Weight": weight,

            "Family_History_Overweight":
                family_history,

            "Frequent_High_Caloric_Food":
                favc,

            "Vegetable_Consumption_Freq":
                fcvc,

            "Main_Meals_Per_Day":
                ncp,

            "Food_Between_Meals":
                caec,

            "Smokes":
                smoke,

            "Daily_Water_Intake":
                ch2o,

            "Calorie_Monitoring":
                scc,

            "Physical_Activity_Freq":
                faf,

            "Technology_Usage_Time":
                tue,

            "Alcohol_Consumption":
                calc,

            "Transportation_Mode":
                mtrans

        }])


        # ====================================================
        # MODEL
        # ====================================================

        pipeline = models[
            chosen_model_name
        ]


        try:

            pred_encoded = pipeline.predict(
                input_row
            )[0]


            pred_label = (
                label_encoder
                .inverse_transform(
                    [pred_encoded]
                )[0]
            )

        except Exception as e:

            st.error(
                "Unable to generate the prediction."
            )

            st.code(
                str(e)
            )

            st.stop()


        # ====================================================
        # BMI
        # ====================================================

        bmi_value = (
            weight / (height ** 2)
        )


        # ====================================================
        # SAVE TO HISTORY
        # ====================================================

        history_record = {

            "Date & Time":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "Model":
                chosen_model_name,

            "Predicted Obesity Level":
                pred_label.replace(
                    "_",
                    " "
                ),

            "Gender":
                gender,

            "Age":
                age,

            "Height (m)":
                round(
                    height,
                    2
                ),

            "Weight (kg)":
                round(
                    weight,
                    1
                ),

            "BMI":
                round(
                    bmi_value,
                    2
                ),

            "Family History":
                family_history,

            "High-Calorie Food":
                favc,

            "Vegetable Frequency":
                fcvc,

            "Main Meals":
                ncp,

            "Food Between Meals":
                caec,

            "Smokes":
                smoke,

            "Water Intake":
                ch2o,

            "Calorie Monitoring":
                scc,

            "Physical Activity":
                faf,

            "Technology Usage":
                tue,

            "Alcohol":
                calc,

            "Transportation":
                mtrans

        }


        st.session_state.prediction_history.insert(
            0,
            history_record
        )


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        result_col, chart_col = st.columns(
            [1, 1.3]
        )


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        with result_col:

            st.subheader(
                "🎯 Prediction Result"
            )


            st.metric(

                "Predicted Obesity Level",

                pred_label.replace(
                    "_",
                    " "
                )
            )


            st.caption(
                f"Model used: **{chosen_model_name}**"
            )


            st.metric(
                "Calculated BMI",
                f"{bmi_value:.2f}"
            )


            st.markdown(
                "**Recommendation:**"
            )


            st.write(
                RECOMMENDATIONS.get(
                    pred_label,
                    "Consult a healthcare professional for guidance."
                )
            )


        # ====================================================
        # CLASS PROBABILITY CHART
        # ====================================================

        with chart_col:

            if hasattr(
                pipeline,
                "predict_proba"
            ):

                try:

                    proba = pipeline.predict_proba(
                        input_row
                    )[0]


                    proba_df = pd.DataFrame({

                        "Obesity_Level":
                            label_encoder.classes_,

                        "Probability":
                            proba

                    })


                    proba_df = (
                        proba_df
                        .set_index(
                            "Obesity_Level"
                        )
                        .reindex(
                            obesity_order
                        )
                        .reset_index()
                    )


                    fig = px.bar(

                        proba_df,

                        x="Probability",

                        y="Obesity_Level",

                        orientation="h",

                        title=(
                            f"Class Probabilities — "
                            f"{chosen_model_name}"
                        ),

                        text="Probability",

                        category_orders={
                            "Obesity_Level":
                                obesity_order
                        }
                    )


                    fig.update_traces(

                        texttemplate="%{x:.1%}",

                        textposition="outside",

                        hovertemplate=(
                            "<b>Obesity Level:</b> %{y}<br>"
                            "<b>Probability:</b> %{x:.2%}"
                            "<extra></extra>"
                        )
                    )


                    fig.update_xaxes(

                        range=[0, 1],

                        title="Predicted Probability",

                        tickformat=".0%"
                    )


                    fig.update_yaxes(
                        title="Obesity Level"
                    )


                    fig.update_layout(

                        height=450,

                        margin=dict(
                            l=20,
                            r=30,
                            t=60,
                            b=40
                        )
                    )


                    st.plotly_chart(

                        fig,

                        use_container_width=True,

                        config={
                            "displayModeBar": False
                        }
                    )

                except Exception:

                    st.info(
                        "Class probabilities could not be displayed."
                    )

            else:

                st.info(
                    "This model does not expose class probabilities."
                )


# ============================================================
# TAB 3 — DATA EXPLORATION
# ============================================================

with tab_explore:

    st.header(
        "📊 Data Exploration"
    )


    df = load_cleaned_data()


    st.caption(
        f"Cleaned dataset: "
        f"{df.shape[0]} rows × {df.shape[1]} columns"
    )


    # ========================================================
    # DATA PREVIEW
    # ========================================================

    with st.expander(
        "Preview raw table & summary statistics",
        expanded=False
    ):

        st.dataframe(
            df.head(20),
            width="stretch"
        )


        st.write(
            "Numeric summary:"
        )


        st.dataframe(
            df.describe().T,
            width="stretch"
        )


    # ========================================================
    # FILTERS
    # ========================================================

    st.subheader(
        "🔎 Filters"
    )


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

            key="gender_filter"
        )


    with fc2:

        level_filter = st.multiselect(

            "Obesity Level",

            obesity_order,

            default=obesity_order,

            key="level_filter"
        )


    with fc3:

        age_range = st.slider(

            "Age range",

            int(
                df["Age"].min()
            ),

            int(
                df["Age"].max()
            ),

            (
                int(
                    df["Age"].min()
                ),

                int(
                    df["Age"].max()
                )
            ),

            key="age_filter"
        )


    filtered = df[

        df["Gender"].isin(
            gender_filter
        )

        &

        df["Obesity_Level"].isin(
            level_filter
        )

        &

        df["Age"].between(
            *age_range
        )
    ]


    st.caption(
        f"Showing {len(filtered)} of {len(df)} records after filtering."
    )


    # ========================================================
    # NUMERIC OPTIONS
    # ========================================================

    num_options = (
        filtered
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )


    # ========================================================
    # DISTRIBUTION OVERVIEW
    # ========================================================

    with st.container(
        border=True
    ):

        st.markdown(
            "## 📊 Distribution Overview"
        )


        chart1, chart2 = st.columns(
            2,
            gap="medium"
        )


        # ====================================================
        # PIE CHART
        # ====================================================

        with chart1:

            with st.container(
                height=700,
                border=True
            ):

                st.markdown(
                    "### 🍩 Obesity Level Distribution"
                )


                counts = (

                    filtered[
                        "Obesity_Level"
                    ]

                    .value_counts()

                    .reindex(
                        obesity_order
                    )

                    .fillna(0)

                    .reset_index()
                )


                counts.columns = [
                    "Obesity_Level",
                    "Count"
                ]


                counts = counts[
                    counts["Count"] > 0
                ]


                fig = px.pie(

                    counts,

                    names="Obesity_Level",

                    values="Count",

                    color="Obesity_Level",

                    color_discrete_map=
                        OBESITY_COLORS,

                    category_orders={
                        "Obesity_Level":
                            obesity_order
                    },

                    hole=0.32
                )


                fig.update_traces(

                    textinfo="percent",

                    textposition="inside",

                    hovertemplate=(
                        "<b>Obesity Level:</b> %{label}<br>"
                        "<b>Count:</b> %{value}<br>"
                        "<b>Percentage:</b> %{percent}"
                        "<extra></extra>"
                    ),

                    domain=dict(
                        x=[0.02, 0.98],
                        y=[0.05, 0.85]
                    )
                )


                fig.update_layout(

                    height=450,

                    margin=dict(
                        l=0,
                        r=0,
                        t=0,
                        b=50
                    ),

                    legend=dict(

                        title="Obesity Level",

                        orientation="h",

                        x=0.5,

                        xanchor="center",

                        y=-0.35,

                        yanchor="top",

                        font=dict(
                            size=10
                        )
                    )
                )


                st.plotly_chart(

                    fig,

                    use_container_width=True,

                    config={
                        "displayModeBar": False
                    }
                )


        # ====================================================
        # HISTOGRAM
        # ====================================================

        with chart2:

            with st.container(
                height=700,
                border=True
            ):

                st.markdown(
                    "### 📊 Numeric Distribution"
                )


                numeric_cols = (

                    filtered

                    .select_dtypes(
                        include=np.number
                    )

                    .columns

                    .tolist()
                )


                if numeric_cols:

                    hist_col = st.selectbox(

                        "Choose a numeric column",

                        numeric_cols,

                        index=(

                            numeric_cols.index(
                                "BMI"
                            )

                            if "BMI" in numeric_cols

                            else 0
                        ),

                        key="hist_col"
                    )


                    hist_bins = st.slider(

                        "Number of bins",

                        min_value=5,

                        max_value=40,

                        value=20,

                        step=5,

                        key="hist_bins"
                    )


                    fig = px.histogram(

                        filtered,

                        x=hist_col,

                        nbins=hist_bins,

                        marginal="box",

                        opacity=0.85,

                        title=f"Distribution of {hist_col}"
                    )


                    fig.update_traces(

                        hovertemplate=(

                            f"<b>{hist_col}</b>: %{{x}}<br>"

                            "<b>Count:</b> %{y}"

                            "<extra></extra>"
                        )
                    )


                    fig.update_layout(

                        height=450,

                        xaxis=dict(

                            title=hist_col,

                            showgrid=False
                        ),

                        yaxis=dict(

                            title="Number of Records",

                            showgrid=True
                        ),

                        bargap=0.05,

                        hovermode="x unified",

                        margin=dict(
                            l=50,
                            r=30,
                            t=60,
                            b=50
                        )
                    )


                    st.plotly_chart(

                        fig,

                        use_container_width=True,

                        config={
                            "displayModeBar": False
                        }
                    )


                    st.caption(
                        "💡 Hover over the bars to view "
                        "the number of records in each range."
                    )


    # ========================================================
    # SCATTER + BOXPLOT
    # ========================================================

    st.divider()


    chart3, chart4 = st.columns(2)


    # ========================================================
    # SCATTERPLOT
    # ========================================================

    with chart3:

        with st.container(
            height=800,
            border=True
        ):

            st.markdown(
                "### 🔵 Interactive Scatterplot"
            )


            sx = st.selectbox(

                "X-axis",

                num_options,

                index=(

                    num_options.index(
                        "Height"
                    )

                    if "Height" in num_options

                    else 0
                ),

                key="scatter_x"
            )


            sy = st.selectbox(

                "Y-axis",

                num_options,

                index=(

                    num_options.index(
                        "Weight"
                    )

                    if "Weight" in num_options

                    else min(
                        1,
                        len(num_options) - 1
                    )
                ),

                key="scatter_y"
            )


            fig = px.scatter(

                filtered,

                x=sx,

                y=sy,

                color="Obesity_Level",

                category_orders={
                    "Obesity_Level":
                        obesity_order
                },

                color_discrete_map=
                    OBESITY_COLORS,

                hover_data=[

                    "Gender",
                    "Age",
                    "Height",
                    "Weight",
                    "BMI",
                    "Obesity_Level"

                ],

                title=f"{sy} vs {sx}"
            )


            fig.update_traces(

                marker=dict(
                    size=8,
                    opacity=0.75
                )
            )


            fig.update_layout(

                height=550,

                legend=dict(
                    title="Obesity Level",
                    traceorder="normal"
                )
            )


            st.plotly_chart(

                fig,

                use_container_width=True
            )


    # ========================================================
    # BOXPLOT
    # ========================================================

    with chart4:

        with st.container(
            height=800,
            border=True
        ):

            st.markdown(
                "### 📦 Interactive Boxplot"
            )


            box_col = st.selectbox(

                "Numeric variable",

                num_options,

                index=(

                    num_options.index(
                        "Weight"
                    )

                    if "Weight" in num_options

                    else 0
                ),

                key="box_col"
            )


            fig = px.box(

                filtered,

                x="Obesity_Level",

                y=box_col,

                color="Obesity_Level",

                category_orders={
                    "Obesity_Level":
                        obesity_order
                },

                color_discrete_map=
                    OBESITY_COLORS,

                points="outliers",

                title=f"{box_col} by Obesity Level"
            )


            fig.update_traces(

                hovertemplate=(

                    "<b>Obesity Level:</b> %{x}<br>"

                    f"<b>{box_col}:</b> %{{y}}"

                    "<extra></extra>"
                )
            )


            fig.update_layout(

                height=550,

                xaxis_tickangle=-45,

                showlegend=False
            )


            st.plotly_chart(

                fig,

                use_container_width=True
            )


    # ========================================================
    # CORRELATION HEATMAP
    # ========================================================

    st.divider()


    st.subheader(
        "🔥 Interactive Correlation Heatmap"
    )


    numeric_data = filtered.select_dtypes(
        include=np.number
    )


    corr = numeric_data.corr()


    fig = px.imshow(

        corr,

        text_auto=".2f",

        aspect="auto",

        title="Correlation Between Numeric Variables"
    )


    fig.update_traces(

        hovertemplate=(

            "<b>Feature X:</b> %{x}<br>"

            "<b>Feature Y:</b> %{y}<br>"

            "<b>Correlation:</b> %{z:.2f}"

            "<extra></extra>"
        )
    )


    fig.update_layout(
        height=650
    )


    st.plotly_chart(

        fig,

        use_container_width=True
    )


    # ========================================================
    # OLAP PIVOT EXPLORER
    # ========================================================

    st.divider()


    st.subheader(
        "📋 OLAP-style Pivot Explorer"
    )


    st.caption(
        "Build your own multidimensional summary table by choosing "
        "row and column dimensions, a numeric measure, and an aggregation."
    )


    categorical_options = (

        filtered

        .select_dtypes(
            exclude=np.number
        )

        .columns

        .tolist()
    )


    p1, p2, p3, p4 = st.columns(4)


    with p1:

        row_dim = st.selectbox(

            "Row dimension",

            categorical_options,

            index=(

                categorical_options.index(
                    "Gender"
                )

                if "Gender" in categorical_options

                else 0
            ),

            key="row_dimension"
        )


    with p2:

        col_dim = st.selectbox(

            "Column dimension",

            categorical_options,

            index=(

                categorical_options.index(
                    "Obesity_Level"
                )

                if "Obesity_Level" in categorical_options

                else 0
            ),

            key="column_dimension"
        )


    with p3:

        measure = st.selectbox(

            "Measure",

            num_options,

            index=(

                num_options.index(
                    "BMI"
                )

                if "BMI" in num_options

                else 0
            ),

            key="pivot_measure"
        )


    with p4:

        agg_func = st.selectbox(

            "Aggregation",

            [
                "mean",
                "count",
                "sum",
                "median"
            ],

            key="pivot_aggregation"
        )


    if row_dim != col_dim:

        pivot = pd.pivot_table(

            filtered,

            index=row_dim,

            columns=col_dim,

            values=measure,

            aggfunc=agg_func,

            margins=True,

            margins_name="All (Roll-up)"
        ).round(2)


        st.dataframe(

            pivot,

            width="stretch"
        )

    else:

        st.warning(
            "Choose two different dimensions for rows and columns."
        )


# ============================================================
# TAB 4 — HISTORY
# ============================================================

with tab_history:

    st.header(
        "🕒 Prediction History"
    )


    st.caption(
        "This tab records predictions made during the current "
        "Streamlit session."
    )


    # ========================================================
    # NO HISTORY
    # ========================================================

    if not st.session_state.prediction_history:

        st.info(
            "No prediction history yet. "
            "Go to the **Prediction** tab and make a prediction "
            "to see it here."
        )


    # ========================================================
    # HISTORY AVAILABLE
    # ========================================================

    else:

        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )


        # ====================================================
        # SUMMARY
        # ====================================================

        st.subheader(
            "📌 History Summary"
        )


        h1, h2, h3 = st.columns(3)


        with h1:

            st.metric(
                "Total Predictions",
                len(history_df)
            )


        with h2:

            st.metric(
                "Most Recent Prediction",
                history_df.iloc[0][
                    "Predicted Obesity Level"
                ]
            )


        with h3:

            st.metric(
                "Models Used",
                history_df["Model"].nunique()
            )


        # ====================================================
        # PREDICTION TABLE
        # ====================================================

        st.subheader(
            "📋 Previous Predictions"
        )


        st.dataframe(

            history_df,

            use_container_width=True,

            hide_index=True
        )


        # ====================================================
        # DOWNLOAD HISTORY
        # ====================================================

        st.subheader(
            "💾 Export History"
        )


        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")


        st.download_button(

            label="⬇️ Download Prediction History (CSV)",

            data=csv_data,

            file_name="prediction_history.csv",

            mime="text/csv",

            use_container_width=True
        )


        # ====================================================
        # HISTORY DISTRIBUTION
        # ====================================================

        st.subheader(
            "📊 Prediction History Distribution"
        )


        history_counts = (

            history_df[
                "Predicted Obesity Level"
            ]

            .value_counts()

            .reset_index()
        )


        history_counts.columns = [
            "Obesity Level",
            "Count"
        ]


        fig_history = px.bar(

            history_counts,

            x="Obesity Level",

            y="Count",

            title="Predicted Obesity Levels",

            text="Count"
        )


        fig_history.update_traces(

            textposition="outside",

            hovertemplate=(

                "<b>Obesity Level:</b> %{x}<br>"

                "<b>Predictions:</b> %{y}"

                "<extra></extra>"
            )
        )


        fig_history.update_layout(

            height=450,

            xaxis_tickangle=-45
        )


        st.plotly_chart(

            fig_history,

            use_container_width=True
        )


        # ====================================================
        # CLEAR HISTORY
        # ====================================================

        st.divider()


        if st.button(

            "🗑️ Clear Prediction History",

            type="secondary",

            use_container_width=True
        ):

            st.session_state.prediction_history = []

            st.rerun()


# ============================================================
# TAB 5 — MODEL PERFORMANCE
# ============================================================

with tab_performance:

    st.header(
        "📈 Model Performance"
    )


    # ========================================================
    # LOAD COMPARISON RESULTS
    # ========================================================

    comparison_df = load_comparison_table()


    if comparison_df.index.name is None:

        comparison_df.index.name = "Model"


    # ========================================================
    # 1. CONSOLIDATED COMPARISON TABLE
    # ========================================================

    st.subheader(
        "Consolidated Comparison Table"
    )


    st.caption(
        "`Train_Accuracy_InSample` is the model scored on the exact "
        "rows it was fit on. It can be very high for flexible models "
        "and should not be used alone as an overfitting diagnostic. "
        "`Train_Accuracy_CV` is based on held-out training folds and "
        "is the more meaningful value for checking generalisation."
    )


    st.dataframe(

        comparison_df.style.format(
            precision=4
        ),

        use_container_width=True
    )


    # ========================================================
    # 2. OUTER COMPARISON
    # ========================================================

    st.subheader(
        "Outer Comparison — Models Against Each Other (Test Set)"
    )


    outer_metrics = [

        "Test_Accuracy",

        "Precision_Weighted",

        "Recall_Weighted",

        "F1_Weighted"

    ]


    missing_outer = [

        col
        for col in outer_metrics
        if col not in comparison_df.columns

    ]


    if missing_outer:

        st.warning(

            f"The comparison table is missing these columns: "
            f"{missing_outer}"
        )

    else:

        outer_plot_df = (

            comparison_df[
                outer_metrics
            ]

            .reset_index()

            .rename(
                columns={
                    "index": "Model"
                }
            )
        )


        outer_long = outer_plot_df.melt(

            id_vars="Model",

            var_name="Metric",

            value_name="Score"
        )


        fig = px.bar(

            outer_long,

            x="Model",

            y="Score",

            color="Metric",

            barmode="group",

            title="Model Performance Comparison"
        )


        fig.update_traces(

            hovertemplate=(

                "<b>Model:</b> %{x}<br>"

                "<b>Metric:</b> %{fullData.name}<br>"

                "<b>Score:</b> %{y:.2%}"

                "<extra></extra>"
            )
        )


        fig.update_yaxes(

            range=[0, 1.05],

            title="Score",

            tickformat=".0%"
        )


        fig.update_layout(

            height=500,

            hovermode="x unified"
        )


        st.plotly_chart(

            fig,

            use_container_width=True
        )


    # ========================================================
    # 3. TRAIN VS TEST ACCURACY
    # ========================================================

    st.subheader(
        "Inner Comparison — Train vs Test Accuracy (Overfitting Check)"
    )


    inner_tab1, inner_tab2 = st.tabs(

        [
            "Cross-validated (trust this)",
            "In-sample (for transparency)"
        ]
    )


    # ========================================================
    # 3A. CROSS VALIDATED
    # ========================================================

    with inner_tab1:

        st.subheader(
            "Inner Comparison — Train vs Test Accuracy"
        )


        accuracy_columns = [

            "Train_Accuracy_CV",

            "Test_Accuracy"

        ]


        missing_accuracy = [

            col
            for col in accuracy_columns
            if col not in comparison_df.columns

        ]


        if missing_accuracy:

            st.warning(

                f"The comparison table is missing: "
                f"{missing_accuracy}"
            )

        else:

            accuracy_df = (

                comparison_df[
                    accuracy_columns
                ]

                .reset_index()

                .rename(
                    columns={
                        "index": "Model"
                    }
                )
            )


            accuracy_long = accuracy_df.melt(

                id_vars="Model",

                var_name="Dataset",

                value_name="Accuracy"
            )


            fig = px.bar(

                accuracy_long,

                x="Model",

                y="Accuracy",

                color="Dataset",

                barmode="group",

                text="Accuracy",

                title="Cross-Validated Train vs Test Accuracy"
            )


            fig.update_traces(

                texttemplate="%{text:.1%}",

                textposition="outside",

                hovertemplate=(

                    "<b>Model:</b> %{x}<br>"

                    "<b>Dataset:</b> %{fullData.name}<br>"

                    "<b>Accuracy:</b> %{y:.2%}"

                    "<extra></extra>"
                )
            )


            fig.update_yaxes(

                range=[0, 1.05],

                title="Accuracy",

                tickformat=".0%"
            )


            fig.update_xaxes(
                title="Model"
            )


            fig.update_layout(
                height=500,
                hovermode="x unified"
            )


            st.plotly_chart(

                fig,

                use_container_width=True
            )


            st.caption(

                "Train accuracy here is cross-validated: each validation "
                "fold contains rows that were not used to fit that fold's "
                "model. This makes it a more meaningful comparison with "
                "the final held-out test accuracy."
            )


    # ========================================================
    # 3B. IN-SAMPLE
    # ========================================================

    with inner_tab2:

        st.subheader(
            "In-sample Train Accuracy vs Test Accuracy"
        )


        accuracy_columns = [

            "Train_Accuracy_InSample",

            "Test_Accuracy"

        ]


        missing_accuracy = [

            col
            for col in accuracy_columns
            if col not in comparison_df.columns

        ]


        if missing_accuracy:

            st.warning(

                f"The comparison table is missing: "
                f"{missing_accuracy}"
            )

        else:

            accuracy_df = (

                comparison_df[
                    accuracy_columns
                ]

                .reset_index()

                .rename(
                    columns={
                        "index": "Model"
                    }
                )
            )


            accuracy_long = accuracy_df.melt(

                id_vars="Model",

                var_name="Dataset",

                value_name="Accuracy"
            )


            fig = px.bar(

                accuracy_long,

                x="Model",

                y="Accuracy",

                color="Dataset",

                barmode="group",

                text="Accuracy",

                title="In-Sample Train vs Test Accuracy"
            )


            fig.update_traces(

                texttemplate="%{text:.1%}",

                textposition="outside",

                hovertemplate=(

                    "<b>Model:</b> %{x}<br>"

                    "<b>Dataset:</b> %{fullData.name}<br>"

                    "<b>Accuracy:</b> %{y:.2%}"

                    "<extra></extra>"
                )
            )


            fig.update_yaxes(

                range=[0, 1.05],

                title="Accuracy",

                tickformat=".0%"
            )


            fig.update_xaxes(
                title="Model"
            )


            fig.update_layout(
                height=500
            )


            st.plotly_chart(

                fig,

                use_container_width=True
            )


            st.caption(

                "In-sample accuracy means the model is evaluated on the "
                "same training rows it was fitted on. It is expected to "
                "be high for flexible models and is shown for transparency."
            )


    # ========================================================
    # 4. WEIGHTED VS MACRO
    # ========================================================

    st.subheader(
        "Weighted vs Macro-averaged Metrics"
    )


    st.caption(

        "Weighted averages account for the number of observations "
        "in each class, while macro averages give every class equal "
        "importance. A larger difference can indicate that performance "
        "varies across obesity classes."
    )


    macro_metric_tabs = st.tabs(

        [
            "Precision",
            "Recall",
            "F1"
        ]
    )


    for tab, metric in zip(

        macro_metric_tabs,

        [
            "Precision",
            "Recall",
            "F1"
        ]
    ):

        with tab:

            weighted_col = (
                f"{metric}_Weighted"
            )

            macro_col = (
                f"{metric}_Macro"
            )


            metric_columns = [

                weighted_col,

                macro_col

            ]


            missing_metric = [

                col
                for col in metric_columns
                if col not in comparison_df.columns

            ]


            if missing_metric:

                st.warning(

                    f"The comparison table is missing: "
                    f"{missing_metric}"
                )

                continue


            metric_df = (

                comparison_df[
                    metric_columns
                ]

                .reset_index()

                .rename(
                    columns={
                        "index": "Model"
                    }
                )
            )


            metric_long = metric_df.melt(

                id_vars="Model",

                var_name="Average",

                value_name="Score"
            )


            metric_long["Average"] = (

                metric_long["Average"]

                .str.replace(

                    f"{metric}_",

                    "",

                    regex=False
                )
            )


            fig = px.bar(

                metric_long,

                x="Model",

                y="Score",

                color="Average",

                barmode="group",

                text="Score",

                title=f"{metric}: Weighted vs Macro"
            )


            fig.update_traces(

                texttemplate="%{text:.1%}",

                textposition="outside",

                hovertemplate=(

                    "<b>Model:</b> %{x}<br>"

                    "<b>Average:</b> %{fullData.name}<br>"

                    f"<b>{metric}:</b> %{{y:.2%}}"

                    "<extra></extra>"
                )
            )


            fig.update_yaxes(

                range=[0, 1.05],

                title=metric,

                tickformat=".0%"
            )


            fig.update_xaxes(
                title="Model"
            )


            fig.update_layout(
                height=450
            )


            st.plotly_chart(

                fig,

                use_container_width=True
            )


    # ========================================================
    # 5. MODEL EVALUATION
    # ========================================================

    st.divider()


    st.subheader(
        "Model Evaluation"
    )


    st.caption(

        "Select a model to inspect its test-set predictions, "
        "confusion matrix, ROC curves, classification report, "
        "and feature importance."
    )


    available_eval_models = list(
        models.keys()
    )


    if not available_eval_models:

        st.error(
            "No trained models are available."
        )

    else:

        default_index = (

            available_eval_models.index(
                best_model_name
            )

            if best_model_name in available_eval_models

            else 0
        )


        eval_model_name = st.selectbox(

            "Choose a model to inspect",

            available_eval_models,

            index=default_index,

            key="eval_model"
        )


        # ====================================================
        # REBUILD TEST SPLIT
        # ====================================================

        X_train, X_test, y_train, y_test = (
            rebuild_test_split(
                metadata
            )
        )


        eval_pipeline = models[
            eval_model_name
        ]


        # ====================================================
        # PREDICTIONS
        # ====================================================

        y_pred = eval_pipeline.predict(
            X_test
        )


        # ====================================================
        # TEST SET METRICS
        # ====================================================

        st.markdown(
            "### Test-Set Metrics"
        )


        if eval_model_name in comparison_df.index:

            selected_row = comparison_df.loc[
                eval_model_name
            ]


            m1, m2, m3, m4 = st.columns(4)


            m1.metric(

                "Test Accuracy",

                f"{selected_row['Test_Accuracy']:.1%}"
            )


            m2.metric(

                "Precision",

                f"{selected_row['Precision_Weighted']:.1%}"
            )


            m3.metric(

                "Recall",

                f"{selected_row['Recall_Weighted']:.1%}"
            )


            m4.metric(

                "F1",

                f"{selected_row['F1_Weighted']:.1%}"
            )


            st.markdown(
                "**Macro-average metrics** "
                "(every class counted equally)"
            )


            m5, m6, m7, m8 = st.columns(4)


            m5.metric(
                "—",
                ""
            )


            m6.metric(

                "Precision",

                f"{selected_row['Precision_Macro']:.1%}"
            )


            m7.metric(

                "Recall",

                f"{selected_row['Recall_Macro']:.1%}"
            )


            m8.metric(

                "F1",

                f"{selected_row['F1_Macro']:.1%}"
            )


        else:

            st.warning(

                f"'{eval_model_name}' was not found "
                "in the comparison table."
            )


        # ====================================================
        # CONFUSION MATRIX + ROC
        # ====================================================

        cm_col, roc_col = st.columns(2)


        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        with cm_col:

            with st.container(
                border=True
            ):

                st.markdown(
                    "### Confusion Matrix"
                )


                actual_classes = list(
                    label_encoder.classes_
                )


                ordinal_indices = [

                    actual_classes.index(
                        cls
                    )

                    for cls in ORDINAL_ORDER

                ]


                cm = confusion_matrix(

                    y_test,

                    y_pred,

                    labels=ordinal_indices
                )


                cm_df = pd.DataFrame(

                    cm,

                    index=ORDINAL_ORDER,

                    columns=ORDINAL_ORDER
                )


                fig_cm = px.imshow(

                    cm_df,

                    text_auto=True,

                    aspect="auto",

                    title=(
                        f"Confusion Matrix — "
                        f"{eval_model_name}"
                    ),

                    labels={

                        "x": "Predicted",

                        "y": "Actual",

                        "color": "Count"

                    }
                )


                fig_cm.update_traces(

                    hovertemplate=(

                        "<b>Actual:</b> %{y}<br>"

                        "<b>Predicted:</b> %{x}<br>"

                        "<b>Count:</b> %{z}"

                        "<extra></extra>"
                    )
                )


                fig_cm.update_xaxes(

                    title="Predicted",

                    tickangle=-45
                )


                fig_cm.update_yaxes(
                    title="Actual"
                )


                fig_cm.update_layout(

                    height=600,

                    margin=dict(

                        l=20,

                        r=20,

                        t=70,

                        b=130

                    )
                )


                st.plotly_chart(

                    fig_cm,

                    use_container_width=True
                )


        # ====================================================
        # ROC CURVES
        # ====================================================

        with roc_col:

            with st.container(
                border=True
            ):

                st.markdown(
                    "### Per-class ROC Curves"
                )


                if hasattr(
                    eval_pipeline,
                    "predict_proba"
                ):

                    actual_classes = list(
                        label_encoder.classes_
                    )


                    ordinal_indices = [

                        actual_classes.index(
                            cls
                        )

                        for cls in ORDINAL_ORDER

                    ]


                    y_proba = (
                        eval_pipeline
                        .predict_proba(
                            X_test
                        )
                    )


                    roc_rows = []


                    for class_name, class_index in zip(

                        ORDINAL_ORDER,

                        ordinal_indices

                    ):

                        y_true_binary = (

                            y_test == class_index

                        ).astype(int)


                        y_score = y_proba[
                            :,
                            class_index
                        ]


                        fpr, tpr, _ = roc_curve(

                            y_true_binary,

                            y_score
                        )


                        roc_auc_value = auc(

                            fpr,

                            tpr
                        )


                        for x, y in zip(
                            fpr,
                            tpr
                        ):

                            roc_rows.append({

                                "False Positive Rate":
                                    x,

                                "True Positive Rate":
                                    y,

                                "Obesity Level":
                                    class_name,

                                "AUC":
                                    roc_auc_value

                            })


                    roc_df = pd.DataFrame(
                        roc_rows
                    )


                    fig_roc = px.line(

                        roc_df,

                        x="False Positive Rate",

                        y="True Positive Rate",

                        color="Obesity Level",

                        category_orders={

                            "Obesity Level":
                                ORDINAL_ORDER

                        },

                        title=(
                            f"ROC Curves — "
                            f"{eval_model_name}"
                        ),

                        hover_data=[
                            "AUC"
                        ]
                    )


                    fig_roc.add_scatter(

                        x=[0, 1],

                        y=[0, 1],

                        mode="lines",

                        name="Random Classifier",

                        line=dict(
                            dash="dot"
                        )
                    )


                    fig_roc.update_xaxes(

                        range=[0, 1],

                        title="False Positive Rate"
                    )


                    fig_roc.update_yaxes(

                        range=[0, 1],

                        title="True Positive Rate"
                    )


                    fig_roc.update_layout(

                        height=600,

                        hovermode="closest",

                        margin=dict(

                            l=20,

                            r=20,

                            t=70,

                            b=80

                        )
                    )


                    st.plotly_chart(

                        fig_roc,

                        use_container_width=True
                    )


                else:

                    st.info(

                        "This model does not expose "
                        "class probabilities for ROC curves."
                    )


        # ====================================================
        # CLASSIFICATION REPORT
        # ====================================================

        with st.expander(

            f"Full Classification Report — "
            f"{eval_model_name}"

        ):

            report = classification_report(

                y_test,

                y_pred,

                target_names=label_encoder.classes_,

                zero_division=0,

                output_dict=True
            )


            report_df = (

                pd.DataFrame(
                    report
                )

                .T

                .round(3)
            )


            st.dataframe(

                report_df,

                use_container_width=True
            )


    # ========================================================
    # RANDOM FOREST FEATURE IMPORTANCE
    # ========================================================

    if (

        "eval_model_name" in locals()

        and

        eval_model_name == "Random Forest"

    ):

        st.divider()


        st.subheader(
            "Random Forest — Feature Importance"
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


            importance_df = pd.DataFrame({

                "Feature":
                    feature_names,

                "Importance":
                    importances

            })


            importance_df = (

                importance_df

                .sort_values(

                    "Importance",

                    ascending=False
                )

                .head(15)

                .sort_values(

                    "Importance",

                    ascending=True
                )
            )


            fig = px.bar(

                importance_df,

                x="Importance",

                y="Feature",

                orientation="h",

                title=(
                    "Top 15 Random Forest "
                    "Feature Importances"
                ),

                text="Importance"
            )


            fig.update_traces(

                texttemplate="%{text:.3f}",

                textposition="outside",

                hovertemplate=(

                    "<b>Feature:</b> %{y}<br>"

                    "<b>Importance:</b> %{x:.3f}"

                    "<extra></extra>"
                )
            )


            fig.update_xaxes(
                title="Feature Importance"
            )


            fig.update_yaxes(
                title="Feature"
            )


            fig.update_layout(
                height=600
            )


            st.plotly_chart(

                fig,

                use_container_width=True
            )


        except Exception as e:

            st.warning(
                "Unable to display Random Forest feature importance."
            )

            st.code(
                str(e)
            )
