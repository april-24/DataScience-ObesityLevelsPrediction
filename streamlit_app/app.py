
"""
BMDS2003 Data Science — Obesity Level Prediction App
=====================================================
A Streamlit deployment prototype for the group project

"Estimation of Obesity Levels Based on Eating Habits and Physical Condition"

Run locally:
    streamlit run app.py
"""

import json
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from notebook_detail_pages import render_notebook_eda_page

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
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
    .hero-card {
        padding: 1.6rem 1.8rem; border-radius: 22px;
        background: linear-gradient(135deg, #fff7ed 0%, #ecfdf5 55%, #eff6ff 100%);
        border: 1px solid rgba(15, 118, 110, .16);
        box-shadow: 0 10px 28px rgba(15, 23, 42, .08); margin-bottom: 1rem;
    }
    .hero-card h2 {margin: 0 0 .45rem 0; color: #0f766e;}
    .hero-card p {margin: 0; color: #334155; font-size: 1.02rem;}
    .team-card {
        min-height: 178px; padding: 1.15rem; border-radius: 18px;
        background: var(--secondary-background-color); border: 1px solid rgba(148, 163, 184, .30);
        box-shadow: 0 6px 18px rgba(15, 23, 42, .06); text-align: center;
    }
    .team-icon {font-size: 2.15rem; margin-bottom: .35rem;}
    .team-name {font-weight: 750; font-size: 1.05rem; color: var(--text-color);}
    .team-role {font-size: .90rem; color: #0f766e; margin-top: .35rem;}
    .insight-card {
        padding: .9rem 1rem; border-radius: 14px; background: rgba(14, 165, 233, .07);
        border-left: 4px solid #0ea5e9; margin: .35rem 0 1rem 0;
    }
    div[role="radiogroup"] {gap: .35rem;}
    div[role="radiogroup"] label {
        background: var(--secondary-background-color); border: 1px solid rgba(148, 163, 184, .32);
        padding: .35rem .7rem; border-radius: 999px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATHS
# ============================================================

MODELS_DIR = Path(__file__).parent / "models"


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
        "increasing physical activity may be beneficial."
    ),

    "Overweight_Level_II": (
        "Your inputs suggest overweight. A structured plan combining dietary "
        "adjustments, increased physical activity, and reduced sedentary time "
        "may be beneficial."
    ),

    "Obesity_Type_I": (
        "Your inputs suggest Class I obesity. Consider consulting a healthcare "
        "provider or dietitian for a supervised weight-management plan."
    ),

    "Obesity_Type_II": (
        "Your inputs suggest Class II obesity. Professional medical guidance "
        "is strongly recommended for a safe and supervised intervention plan."
    ),

    "Obesity_Type_III": (
        "Your inputs suggest Class III obesity. Please consult a healthcare "
        "professional to discuss a comprehensive and medically supervised "
        "management plan."
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


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "prediction_reset_counter" not in st.session_state:
    st.session_state.prediction_reset_counter = 0


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource(show_spinner="Loading trained models...")
def load_models():

    models = {}

    model_errors = {}

    for name, filename in MODEL_FILES.items():

        path = MODELS_DIR / filename

        if not path.exists():

            model_errors[name] = (
                f"File not found: {path}"
            )

            continue

        try:

            models[name] = joblib.load(path)

        except Exception as e:

            model_errors[name] = str(e)

    # Label encoder
    label_encoder_path = MODELS_DIR / "label_encoder.pkl"

    if not label_encoder_path.exists():

        raise FileNotFoundError(
            f"Missing label encoder: {label_encoder_path}"
        )

    try:

        label_encoder = joblib.load(
            label_encoder_path
        )

    except Exception as e:

        raise RuntimeError(
            "Unable to load label_encoder.pkl. "
            "This usually indicates an incompatible "
            "Python/scikit-learn/joblib environment.\n\n"
            f"Original error: {e}"
        )

    return models, label_encoder, model_errors


# ============================================================
# LOAD METADATA
# ============================================================

@st.cache_resource(show_spinner=False)
def load_metadata():

    path = MODELS_DIR / "feature_metadata.json"

    if not path.exists():

        raise FileNotFoundError(
            f"Missing metadata file: {path}"
        )

    with open(path, "r") as f:

        return json.load(f)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(show_spinner="Loading dataset...")
def load_cleaned_data():

    path = MODELS_DIR / "obesity_cleaned.csv"

    if not path.exists():

        raise FileNotFoundError(
            f"Missing dataset: {path}"
        )

    return pd.read_csv(path)


# ============================================================
# LOAD MODEL COMPARISON
# ============================================================

@st.cache_data(show_spinner=False)
def load_comparison_table():

    path = MODELS_DIR / "model_comparison.csv"

    if not path.exists():

        raise FileNotFoundError(
            f"Missing comparison file: {path}"
        )

    return pd.read_csv(
        path,
        index_col=0
    )


# ============================================================
# REBUILD TEST SPLIT
# ============================================================

@st.cache_data(
    show_spinner="Rebuilding the held-out test split..."
)
def rebuild_test_split(_metadata):

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

try:

    models, label_encoder, model_errors = load_models()

    metadata = load_metadata()

except Exception as e:

    st.error(
        "❌ The application could not load the saved models."
    )

    st.code(
        str(e)
    )

    st.warning(
        """
This is usually caused by the model files being created with a
different Python/scikit-learn/joblib version from the one used by
Streamlit Cloud.

Make sure the environment used to train the models matches the
environment used to deploy them.
"""
    )

    st.stop()


# ============================================================
# MODEL ERROR WARNING
# ============================================================

if model_errors:

    st.warning(
        "Some models could not be loaded."
    )

    for model_name, error in model_errors.items():

        st.error(
            f"**{model_name}**\n\n{error}"
        )


# ============================================================
# METADATA
# ============================================================

obesity_order = metadata.get(
    "obesity_order",
    [
        "Insufficient_Weight",
        "Normal_Weight",
        "Overweight_Level_I",
        "Overweight_Level_II",
        "Obesity_Type_I",
        "Obesity_Type_II",
        "Obesity_Type_III"
    ]
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
# HORIZONTAL PAGE NAVIGATION
# ============================================================

PAGE_ABOUT = "🏠 About"
PAGE_EXPLORE = "📊 Data Exploration"
PAGE_EDA_GALLERY = "🧭 More EDA"
PAGE_PREDICT = "🔮 Prediction"
PAGE_HISTORY = "🕒 History"
PAGE_PERFORMANCE = "📈 Model Evaluation"
PAGE_RESULTS = "🔬 Detailed Results"

NAVIGATION_OPTIONS = [
    PAGE_ABOUT,
    PAGE_EXPLORE,
    PAGE_PREDICT,
    PAGE_HISTORY,
    PAGE_EDA_GALLERY,
    PAGE_PERFORMANCE,
    PAGE_RESULTS,
]


def go_to_page(page_name):
    """Navigation callback used by the in-page 'view more' buttons."""
    st.session_state["main_navigation"] = page_name


active_page = st.radio(
    "Main navigation",
    NAVIGATION_OPTIONS,
    horizontal=True,
    label_visibility="collapsed",
    key="main_navigation",
)


# ============================================================
# TAB 1 — ABOUT
# ============================================================

if active_page == PAGE_ABOUT:

    st.markdown(
        """
        <div class="hero-card">
            <h2>🍎 Obesity Level Prediction</h2>
            <p>An interactive BMDS2003 data-science project that connects lifestyle,
            eating habits and physical condition with seven obesity-level categories.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(
        [2, 1]
    )

    # --------------------------------------------------------
    # LEFT
    # --------------------------------------------------------

    with col1:

        st.subheader(
            "Business Understanding"
        )

        st.markdown(
            """
Obesity is a growing public-health concern linked to diabetes,
cardiovascular disease, and reduced quality of life.

This project explores whether **everyday lifestyle and eating
habits can be used to predict a person's obesity category**.

The system can support:

- **Self-assessment**
- **Public-health screening**
- **Lifestyle analysis**
- **Data-driven obesity classification**

**Objective:** classify an individual into one of seven
obesity levels using lifestyle, dietary, and physical-condition
attributes.
"""
        )

        st.subheader(
            "Dataset"
        )

        st.markdown(
            """
- **Source:** UCI Machine Learning Repository
- **Source records:** 2,111 respondents
- **Cleaned modelling records:** 2,087
- **Features:** 16 predictive features
- **Target:** `Obesity_Level`
- **Classes:** 7 obesity categories
"""
        )

        st.subheader(
            "CRISP-DM Workflow"
        )

        st.markdown(
            """
1. **Business Understanding**
2. **Data Understanding**
3. **Data Preparation**
4. **Modelling**
5. **Evaluation**
6. **Deployment**
"""
        )

    # --------------------------------------------------------
    # RIGHT
    # --------------------------------------------------------

    with col2:

        st.subheader("Meet the Team")

        team_members = [
            ("🌳", "Kyra Aerin Leong", "Decision Tree"),
            ("🌲", "Low Li Ping", "Random Forest"),
            ("📐", "Wong Wen Hsuan", "Support Vector Machine"),
            ("🧭", "Gladys Lee", "K-Nearest Neighbours"),
        ]

        for icon, full_name, model_name in team_members:
            st.markdown(
                f"""
                <div class="team-card">
                    <div class="team-icon">{icon}</div>
                    <div class="team-name">{full_name}</div>
                    <div class="team-role">Model focus: {model_name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")

        st.subheader(
            "Best Model"
        )

        st.success(
            f"**{best_model_name}**"
        )

        try:

            comparison_preview = (
                load_comparison_table()
            )

            if best_model_name in comparison_preview.index:

                st.metric(
                    "Best Test Accuracy",
                    f"{comparison_preview.loc[best_model_name, 'Test_Accuracy']:.1%}"
                )

        except Exception:

            st.info(
                "Comparison table unavailable."
            )


# ============================================================
# TAB 2 — PREDICTION
# ============================================================

if active_page == PAGE_PREDICT:

    st.header(
        "🔮 Predict an Obesity Level"
    )

    st.caption(
        "Fill in the fields below and choose a model to generate "
        "a live prediction."
    )

    num_meta = metadata[
        "numeric_features"
    ]

    cat_meta = metadata[
        "categorical_features"
    ]


    # ========================================================
    # FREQUENCY
    # ========================================================

    FREQUENCY_ORDER = [
        "no",
        "Sometimes",
        "Frequently",
        "Always"
    ]


    def freq_label(x):

        return (
            "Never"
            if x == "no"
            else x
        )


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
            "No trained models are available."
        )

        st.stop()


    default_model_index = (

        available_models.index(
            best_model_name
        )

        if best_model_name in available_models

        else 0

    )


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
    # FORM
    # ========================================================

    with st.form(
        f"prediction_form_{reset_counter}"
    ):

        # ----------------------------------------------------
        # PERSONAL
        # ----------------------------------------------------

        st.subheader(
            "Personal & Physical Attributes"
        )

        c1, c2, c3 = st.columns(3)


        with c1:

            gender = st.radio(

                "Gender",

                cat_meta["Gender"],

                horizontal=True,

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
                ) + 50,

                value=round(
                    num_meta["Weight"]["mean"],
                    1
                ),

                step=1.0,

                key=f"weight_{reset_counter}"

            )


        # ----------------------------------------------------
        # EATING HABITS
        # ----------------------------------------------------

        st.subheader(
            "Eating Habits"
        )

        c4, c5, c6 = st.columns(3)


        with c4:

            family_history = st.radio(

                "Family history of overweight?",

                cat_meta[
                    "Family_History_Overweight"
                ],

                horizontal=True,

                format_func=yes_no_label,

                key=f"family_history_{reset_counter}"

            )


            favc = st.radio(

                "Frequently eats high-caloric food?",

                cat_meta[
                    "Frequent_High_Caloric_Food"
                ],

                horizontal=True,

                format_func=yes_no_label,

                key=f"favc_{reset_counter}"

            )


        with c5:

            fcvc = st.slider(

                "Vegetable consumption frequency",

                1.0,

                3.0,

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

                1.0,

                4.0,

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

            caec = st.select_slider(

                "Eats food between meals?",

                options=FREQUENCY_ORDER,

                value="Sometimes",

                format_func=freq_label,

                key=f"caec_{reset_counter}"

            )


            calc = st.select_slider(

                "Alcohol consumption",

                options=FREQUENCY_ORDER,

                value="Sometimes",

                format_func=freq_label,

                key=f"calc_{reset_counter}"

            )


        # ----------------------------------------------------
        # LIFESTYLE
        # ----------------------------------------------------

        st.subheader(
            "Lifestyle & Physical Condition"
        )

        c7, c8, c9 = st.columns(3)


        with c7:

            smoke = st.radio(

                "Smokes?",

                cat_meta["Smokes"],

                horizontal=True,

                format_func=yes_no_label,

                key=f"smoke_{reset_counter}"

            )


            scc = st.radio(

                "Monitors calorie intake?",

                cat_meta["Calorie_Monitoring"],

                horizontal=True,

                format_func=yes_no_label,

                key=f"scc_{reset_counter}"

            )


        with c8:

            ch2o = st.slider(

                "Daily water intake",

                1.0,

                3.0,

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

                "Physical activity frequency",

                0.0,

                3.0,

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

                "Technology usage time",

                0.0,

                2.0,

                value=round(
                    num_meta[
                        "Technology_Usage_Time"
                    ]["mean"],
                    1
                ),

                step=0.1,

                key=f"tue_{reset_counter}"

            )


            mtrans = st.pills(

                "Usual transportation mode",

                cat_meta[
                    "Transportation_Mode"
                ],

                default=cat_meta[
                    "Transportation_Mode"
                ][0],

                format_func=lambda x:
                    x.replace("_", " "),

                key=f"mtrans_{reset_counter}"

            )


            if mtrans is None:

                mtrans = cat_meta[
                    "Transportation_Mode"
                ][0]


        # ----------------------------------------------------
        # BUTTONS
        # ----------------------------------------------------

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
                "Prediction failed."
            )

            st.code(
                str(e)
            )

            st.stop()


        # ====================================================
        # BMI
        # ====================================================

        bmi_value = (
            weight /
            (height ** 2)
        )


        # ====================================================
        # SAVE HISTORY
        # ====================================================

        history_record = {

            "Model":
                chosen_model_name,

            "Age":
                age,

            "Gender":
                gender,

            "Height":
                height,

            "Weight":
                weight,

            "BMI":
                round(
                    bmi_value,
                    2
                ),

            "Predicted Obesity Level":
                pred_label.replace(
                    "_",
                    " "
                )

        }


        st.session_state.prediction_history.append(
            history_record
        )


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        result_col, chart_col = st.columns(
            [1, 1.3]
        )


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
        # PROBABILITY CHART
        # ====================================================

        with chart_col:

            if hasattr(
                pipeline,
                "predict_proba"
            ):

                try:

                    proba = (
                        pipeline
                        .predict_proba(
                            input_row
                        )[0]
                    )

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

                        color="Obesity_Level",

                        color_discrete_map=
                            OBESITY_COLORS,

                        category_orders={

                            "Obesity_Level":
                                obesity_order

                        },

                        title=(
                            f"Class Probabilities — "
                            f"{chosen_model_name}"
                        ),

                        text="Probability"

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

                        height=500,

                        margin=dict(
                            l=20,
                            r=30,
                            t=60,
                            b=40
                        ),

                        showlegend=False

                    )


                    st.plotly_chart(

                        fig,

                        use_container_width=True,

                        config={
                            "displayModeBar": False
                        }

                    )

                except Exception as e:

                    st.warning(
                        f"Unable to display probabilities: {e}"
                    )

            else:

                st.info(
                    "This model does not expose class probabilities."
                )


# ============================================================
# TAB 3 — DATA EXPLORATION
# ============================================================

if active_page == PAGE_EXPLORE:

    st.header(
        "📊 Data Exploration"
    )

    df = load_cleaned_data()

    st.caption(
        f"Cleaned dataset: "
        f"{df.shape[0]} rows × {df.shape[1]} columns"
    )

    st.button(
        "🧭 Open the additional notebook EDA",
        on_click=go_to_page,
        args=(PAGE_EDA_GALLERY,),
        help="View the meaningful notebook charts that are not reproduced by the dropdown explorers below.",
    )


    # ========================================================
    # PREVIEW
    # ========================================================

    with st.expander(
        "Preview raw table & summary statistics"
    ):

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

        st.write(
            "Numeric summary:"
        )

        st.dataframe(
            df.describe().T,
            use_container_width=True
        )


    # ========================================================
    # FILTERS
    # ========================================================

    st.markdown('<div id="global-filters"></div>', unsafe_allow_html=True)

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

            int(df["Age"].min()),

            int(df["Age"].max()),

            (
                int(df["Age"].min()),
                int(df["Age"].max())
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
        f"Showing {len(filtered)} of "
        f"{len(df)} records after filtering."
    )


    num_options = (
        filtered
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )


    # ========================================================
    # DISTRIBUTION
    # ========================================================

    with st.container(
        border=True
    ):

        st.markdown(
            "## 📊 Distribution Overview"
        )

        chart1, chart2 = st.columns(2)


        # ----------------------------------------------------
        # PIE
        # ----------------------------------------------------

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

        st.markdown("[⬆ Change the global filters](#global-filters)")


        # ----------------------------------------------------
        # HISTOGRAM
        # ----------------------------------------------------

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


                hist_col = st.selectbox(

                    "Choose a numeric column",

                    numeric_cols,

                    index=(

                        numeric_cols.index("BMI")

                        if "BMI" in numeric_cols

                        else 0

                    ),

                    key="hist_col"

                )


                hist_bins = st.slider(

                    "Number of bins",

                    5,

                    40,

                    20,

                    5,

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


                fig.update_layout(
                    height=450
                )


                st.plotly_chart(

                    fig,

                    use_container_width=True,

                    config={
                        "displayModeBar": False
                    }

                )


    # ========================================================
    # SCATTER + BOX
    # ========================================================

    st.divider()

    chart3, chart4 = st.columns(2)


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

                    num_options.index("Height")

                    if "Height" in num_options

                    else 0

                ),

                key="scatter_x"

            )


            sy = st.selectbox(

                "Y-axis",

                num_options,

                index=(

                    num_options.index("Weight")

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
                height=550
            )


            st.plotly_chart(

                fig,

                use_container_width=True

            )


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

                    num_options.index("Weight")

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


            fig.update_layout(

                height=550,

                xaxis_tickangle=-45,

                showlegend=False

            )


            st.plotly_chart(

                fig,

                use_container_width=True

            )

    st.markdown("[⬆ Change the global filters](#global-filters)")


    # ========================================================
    # HEATMAP
    # ========================================================

    st.divider()

    st.subheader(
        "🔥 Interactive Correlation Heatmap"
    )


    numeric_data = (
        filtered
        .select_dtypes(
            include=np.number
        )
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

    st.markdown("[⬆ Change the global filters](#global-filters)")


    # ========================================================
    # OLAP
    # ========================================================

    st.divider()

    st.subheader(
        "📋 OLAP-style Pivot Explorer"
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

                if "Obesity_Level"
                in categorical_options

                else 0

            ),

            key="column_dimension"

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

            use_container_width=True

        )

        st.markdown("[⬆ Change the global filters](#global-filters)")

    else:

        st.warning(
            "Choose two different dimensions."
        )


# ============================================================
# PAGE — ADDITIONAL NOTEBOOK EDA
# ============================================================

if active_page == PAGE_EDA_GALLERY:
    render_notebook_eda_page(
        df=load_cleaned_data(),
        obesity_order=obesity_order,
        obesity_colors=OBESITY_COLORS,
        metadata=metadata,
        go_to_page=go_to_page,
        explore_page_name=PAGE_EXPLORE,
    )


# ============================================================
# TAB 4 — HISTORY
# ============================================================

if active_page == PAGE_HISTORY:

    st.header(
        "🕒 Prediction History"
    )

    st.caption(
        "Your prediction results from this session are displayed here."
    )


    # ========================================================
    # NO HISTORY
    # ========================================================

    if not st.session_state.prediction_history:

        st.info(
            "No predictions have been made yet. "
            "Go to the **Prediction** tab to generate a prediction."
        )


    else:

        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )


        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        h1, h2, h3 = st.columns(3)


        with h1:

            st.metric(
                "Total Predictions",
                len(history_df)
            )


        with h2:

            st.metric(
                "Latest Prediction",
                history_df.iloc[-1][
                    "Predicted Obesity Level"
                ]
            )


        with h3:

            st.metric(
                "Models Used",
                history_df["Model"].nunique()
            )


        st.divider()


        # ----------------------------------------------------
        # HISTORY TABLE
        # ----------------------------------------------------

        st.subheader(
            "📋 Prediction Records"
        )


        st.dataframe(

            history_df,

            use_container_width=True,

            hide_index=True

        )


        # ----------------------------------------------------
        # DISTRIBUTION
        # ----------------------------------------------------

        st.subheader(
            "📊 Prediction Distribution"
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

            color="Obesity Level",

            color_discrete_map=
                OBESITY_COLORS,

            title="Your Prediction History"

        )


        fig_history.update_layout(
            height=450
        )


        st.plotly_chart(

            fig_history,

            use_container_width=True

        )


        # ----------------------------------------------------
        # CLEAR HISTORY
        # ----------------------------------------------------

        st.divider()

        if st.button(
            "🗑️ Clear Prediction History",
            type="secondary"
        ):

            st.session_state.prediction_history = []

            st.rerun()


# ============================================================
# TAB 5 — MODEL PERFORMANCE
# ============================================================

if active_page == PAGE_PERFORMANCE:

    st.header(
        "📈 Model Performance"
    )

    st.button(
        "🔬 Open detailed model diagnostics",
        on_click=go_to_page,
        args=(PAGE_RESULTS,),
        help="View additional notebook-style diagnostics without repeating the charts on this page.",
    )


    # ========================================================
    # LOAD COMPARISON
    # ========================================================

    try:

        comparison_df = (
            load_comparison_table()
        )

    except Exception as e:

        st.error(
            "Unable to load model comparison table."
        )

        st.code(
            str(e)
        )

        st.stop()


    if comparison_df.index.name is None:

        comparison_df.index.name = "Model"


    # ========================================================
    # CONSOLIDATED TABLE
    # ========================================================

    st.subheader(
        "Consolidated Comparison Table"
    )


    st.caption(

        "`Train_Accuracy` is measured on the same rows used for fitting. "
        "`CV_Mean_Accuracy` is averaged across held-out training folds, "
        "while `Test_Accuracy` is measured once on the final test set."

    )


    st.dataframe(

        comparison_df.style.format(
            precision=4
        ),

        use_container_width=True

    )


    # ========================================================
    # OUTER COMPARISON
    # ========================================================

    st.subheader(
        "Outer Comparison — Test Set"
    )


    outer_metrics = [

        "Test_Accuracy",

        "Precision_Macro",

        "Recall_Macro",

        "F1_Macro"

    ]


    missing_outer = [

        col

        for col in outer_metrics

        if col not in comparison_df.columns

    ]


    if missing_outer:

        st.warning(
            f"Missing columns: {missing_outer}"
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

        outer_long["Metric"] = outer_long["Metric"].replace({
            "Test_Accuracy": "Test accuracy",
            "Precision_Macro": "Macro precision",
            "Recall_Macro": "Macro recall",
            "F1_Macro": "Macro F1",
        })


        fig = px.bar(

            outer_long,

            x="Model",

            y="Score",

            color="Metric",

            barmode="group",

            title="Notebook Outer Comparison — Test Accuracy and Macro Metrics"

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
            height=500
        )


        st.plotly_chart(

            fig,

            use_container_width=True

        )


    # ========================================================
    # TRAIN VS TEST
    # ========================================================

    st.subheader(
        "Inner Comparison — Train vs Test Accuracy"
    )


    inner_tab1, inner_tab2 = st.tabs(

        [
            "Cross-validated",
            "In-sample"
        ]

    )


    # --------------------------------------------------------
    # CV
    # --------------------------------------------------------

    with inner_tab1:

        accuracy_columns = [

            "CV_Mean_Accuracy",

            "Test_Accuracy"

        ]


        if all(

            col in comparison_df.columns

            for col in accuracy_columns

        ):

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

            accuracy_long["Dataset"] = accuracy_long["Dataset"].replace({
                "CV_Mean_Accuracy": "CV mean accuracy",
                "Test_Accuracy": "Test accuracy",
            })


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

                textposition="outside"

            )


            fig.update_yaxes(

                range=[0, 1.05],

                tickformat=".0%"

            )


            fig.update_layout(
                height=500
            )


            st.plotly_chart(

                fig,

                use_container_width=True

            )


            st.caption(

                "Cross-validated training accuracy is a more "
                "meaningful comparison with the final held-out "
                "test accuracy."

            )


    # --------------------------------------------------------
    # IN SAMPLE
    # --------------------------------------------------------

    with inner_tab2:

        accuracy_columns = [

            "Train_Accuracy",

            "Test_Accuracy"

        ]


        if all(

            col in comparison_df.columns

            for col in accuracy_columns

        ):

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

            accuracy_long["Dataset"] = accuracy_long["Dataset"].replace({
                "Train_Accuracy": "Training accuracy",
                "Test_Accuracy": "Test accuracy",
            })


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

                textposition="outside"

            )


            fig.update_yaxes(

                range=[0, 1.05],

                tickformat=".0%"

            )


            fig.update_layout(
                height=500
            )


            st.plotly_chart(

                fig,

                use_container_width=True

            )


    # ========================================================
    # WEIGHTED VS MACRO
    # ========================================================

    st.subheader(
        "Weighted vs Macro-averaged Metrics"
    )


    st.caption(

        "Weighted averages account for class frequency, while "
        "macro averages give every obesity class equal importance."

    )


    macro_tabs = st.tabs(

        [
            "Precision",
            "Recall",
            "F1"
        ]

    )


    for tab, metric in zip(

        macro_tabs,

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


            if (

                weighted_col
                not in comparison_df.columns

                or

                macro_col
                not in comparison_df.columns

            ):

                st.warning(
                    "Required columns are unavailable."
                )

                continue


            metric_df = (

                comparison_df[
                    [
                        weighted_col,
                        macro_col
                    ]
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

                textposition="outside"

            )


            fig.update_yaxes(

                range=[0, 1.05],

                tickformat=".0%"

            )


            fig.update_layout(
                height=450
            )


            st.plotly_chart(

                fig,

                use_container_width=True

            )


    # ========================================================
    # MODEL EVALUATION
    # ========================================================

    st.divider()

    st.subheader(
        "Model Evaluation"
    )


    eval_available_models = list(
        models.keys()
    )


    if not eval_available_models:

        st.error(
            "No trained models are available."
        )

    else:

        default_index = (

            eval_available_models.index(
                best_model_name
            )

            if best_model_name
            in eval_available_models

            else 0

        )


        eval_model_name = st.selectbox(

            "Choose a model to inspect",

            eval_available_models,

            index=default_index,

            key="eval_model"

        )


        # ====================================================
        # TEST DATA
        # ====================================================

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = rebuild_test_split(
            metadata
        )


        eval_pipeline = models[
            eval_model_name
        ]


        try:

            y_pred = eval_pipeline.predict(
                X_test
            )

        except Exception as e:

            st.error(
                "Unable to generate test-set predictions."
            )

            st.code(
                str(e)
            )

            st.stop()


        # ====================================================
        # METRICS
        # ====================================================

        st.markdown(
            "### Test-Set Metrics"
        )


        if eval_model_name in comparison_df.index:

            selected_row = (
                comparison_df.loc[
                    eval_model_name
                ]
            )


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
                "**Macro-average metrics**"
            )


            m5, m6, m7 = st.columns(3)


            m5.metric(

                "Precision",

                f"{selected_row['Precision_Macro']:.1%}"

            )


            m6.metric(

                "Recall",

                f"{selected_row['Recall_Macro']:.1%}"

            )


            m7.metric(

                "F1",

                f"{selected_row['F1_Macro']:.1%}"

            )


        # ====================================================
        # CONFUSION MATRIX + ROC
        # ====================================================

        cm_col, roc_col = st.columns(2)


        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        with cm_col:

            with st.container(
                border=True
            ):

                st.markdown(
                    "### Confusion Matrix"
                )


                ordinal_order = [

                    "Insufficient_Weight",

                    "Normal_Weight",

                    "Overweight_Level_I",

                    "Overweight_Level_II",

                    "Obesity_Type_I",

                    "Obesity_Type_II",

                    "Obesity_Type_III"

                ]


                actual_classes = list(
                    label_encoder.classes_
                )


                ordinal_indices = [

                    actual_classes.index(
                        cls
                    )

                    for cls in ordinal_order

                    if cls in actual_classes

                ]


                cm = confusion_matrix(

                    y_test,

                    y_pred,

                    labels=ordinal_indices

                )


                cm_df = pd.DataFrame(

                    cm,

                    index=ordinal_order,

                    columns=ordinal_order

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


                fig_cm.update_xaxes(
                    tickangle=-45
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


        # ----------------------------------------------------
        # ROC
        # ----------------------------------------------------

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

                    try:

                        y_proba = (
                            eval_pipeline
                            .predict_proba(
                                X_test
                            )
                        )


                        roc_rows = []


                        for class_name, class_index in zip(

                            ordinal_order,

                            ordinal_indices

                        ):

                            y_true_binary = (

                                y_test
                                == class_index

                            ).astype(int)


                            y_score = (
                                y_proba[
                                    :,
                                    class_index
                                ]
                            )


                            fpr, tpr, _ = (
                                roc_curve(
                                    y_true_binary,
                                    y_score
                                )
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
                                    ordinal_order

                            },

                            title=(
                                f"ROC Curves — "
                                f"{eval_model_name}"
                            ),

                            hover_data=["AUC"]

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
                            height=600
                        )


                        st.plotly_chart(

                            fig_roc,

                            use_container_width=True

                        )


                    except Exception as e:

                        st.warning(
                            f"Unable to generate ROC curves: {e}"
                        )

                else:

                    st.info(
                        "This model does not expose class probabilities."
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

                target_names=
                    label_encoder.classes_,

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


        # ====================================================
        # RANDOM FOREST FEATURE IMPORTANCE
        # ====================================================

        if eval_model_name == "Random Forest":

            st.divider()

            st.subheader(
                "🌲 Random Forest — Feature Importance"
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

                    textposition="outside"

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


# ============================================================
# PAGE — DETAILED MODEL RESULTS
# ============================================================

if active_page == PAGE_RESULTS:

    st.header("🔬 Detailed Model Results")
    st.caption(
        "Additional diagnostics derived from the notebook artifacts. These figures "
        "complement the Model Evaluation page without repeating its consolidated table, "
        "four-metric comparison, confusion matrices or ROC curves."
    )
    st.button(
        "← Return to Model Evaluation",
        on_click=go_to_page,
        args=(PAGE_PERFORMANCE,),
    )

    try:
        details_df = load_comparison_table()
    except Exception as error:
        st.error("Unable to load the saved model-comparison results.")
        st.code(str(error))
        st.stop()

    if details_df.index.name is None:
        details_df.index.name = "Model"

    best_test_model = details_df["Test_Accuracy"].idxmax()
    best_cv_model = details_df["CV_Macro_F1"].idxmax()
    smallest_gap_model = details_df["Overfit_Gap"].idxmin()
    summary_1, summary_2, summary_3 = st.columns(3)
    summary_1.metric("Highest test accuracy", best_test_model, f"{details_df.loc[best_test_model, 'Test_Accuracy']:.2%}")
    summary_2.metric("Highest CV macro F1", best_cv_model, f"{details_df.loc[best_cv_model, 'CV_Macro_F1']:.2%}")
    summary_3.metric("Smallest train–test gap", smallest_gap_model, f"{details_df.loc[smallest_gap_model, 'Overfit_Gap']:.4f}")

    st.subheader("1. Cross-validation performance and stability")
    cv_chart_df = details_df.reset_index()
    if "Model" not in cv_chart_df.columns:
        cv_chart_df = cv_chart_df.rename(columns={cv_chart_df.columns[0]: "Model"})
    fig_cv = px.bar(
        cv_chart_df,
        x="Model",
        y="CV_Macro_F1",
        error_y="CV_Std_Macro_F1",
        color="Model",
        text="CV_Macro_F1",
        title="Five-fold CV macro F1 with ±1 standard deviation",
    )
    fig_cv.update_traces(
        texttemplate="%{text:.4f}", textposition="outside",
        hovertemplate="<b>%{x}</b><br>Mean CV macro F1: %{y:.4f}<extra></extra>",
    )
    fig_cv.update_yaxes(range=[0, 1.05], tickformat=".0%")
    fig_cv.update_layout(height=480, showlegend=False)
    st.plotly_chart(fig_cv, use_container_width=True)
    st.markdown(
        '<div class="insight-card"><b>What it means:</b> Macro F1 gives every obesity class '
        'equal weight. Taller bars indicate stronger validation performance; shorter error bars '
        'mean the result changes less across folds. Model selection uses this validation evidence '
        'rather than selecting from the test set alone.</div>',
        unsafe_allow_html=True,
    )

    st.subheader("2. Generalisation and overfitting diagnostic")
    gap_df = details_df.reset_index()
    if "Model" not in gap_df.columns:
        gap_df = gap_df.rename(columns={gap_df.columns[0]: "Model"})
    fig_gap = px.scatter(
        gap_df,
        x="Overfit_Gap",
        y="Test_Accuracy",
        color="Model",
        size="Generalisation_Ratio",
        text="Model",
        hover_data={
            "Train_Accuracy": ":.4f",
            "CV_Mean_Accuracy": ":.4f",
            "Test_Accuracy": ":.4f",
            "Overfit_Gap": ":.4f",
            "Generalisation_Ratio": ":.4f",
        },
        title="Test performance versus train–test overfit gap",
    )
    fig_gap.update_traces(textposition="top center", marker=dict(opacity=.82, line=dict(width=1, color="white")))
    fig_gap.update_xaxes(title="Overfit gap (lower is better)")
    fig_gap.update_yaxes(title="Test accuracy (higher is better)", tickformat=".0%")
    fig_gap.update_layout(height=520, showlegend=False)
    st.plotly_chart(fig_gap, use_container_width=True)
    st.markdown(
        '<div class="insight-card"><b>What it means:</b> The preferred region is the upper-left: '
        'high test accuracy with a small train–test gap. A perfect training score is not automatically '
        'desirable when the corresponding test performance falls substantially.</div>',
        unsafe_allow_html=True,
    )

    st.subheader("3. Specificity and probability-ranking quality")
    diagnostic_metrics = ["Specificity_Macro", "AUC_Macro_OVR", "AUC_Weighted_OVR"]
    diagnostic_long = (
        details_df[diagnostic_metrics]
        .reset_index()
        .rename(columns={details_df.index.name or "index": "Model"})
        .melt(id_vars="Model", var_name="Diagnostic", value_name="Score")
    )
    diagnostic_long["Diagnostic"] = diagnostic_long["Diagnostic"].replace({
        "Specificity_Macro": "Macro specificity",
        "AUC_Macro_OVR": "Macro ROC AUC (OvR)",
        "AUC_Weighted_OVR": "Weighted ROC AUC (OvR)",
    })
    fig_diagnostic = px.bar(
        diagnostic_long,
        x="Model",
        y="Score",
        color="Diagnostic",
        barmode="group",
        text="Score",
        title="Additional test-set diagnostics",
    )
    fig_diagnostic.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_diagnostic.update_yaxes(range=[0, 1.05], tickformat=".0%")
    fig_diagnostic.update_layout(height=500, legend=dict(orientation="h", y=-0.22, x=.5, xanchor="center"), margin=dict(b=100))
    st.plotly_chart(fig_diagnostic, use_container_width=True)
    st.markdown(
        '<div class="insight-card"><b>What it means:</b> Macro specificity measures how well '
        'the model rejects each class when that class is not the true label. One-vs-rest AUC evaluates '
        'how well predicted probabilities rank each class above the alternatives across all possible '
        'thresholds. These measures add information beyond the final hard-label accuracy.</div>',
        unsafe_allow_html=True,
    )

    st.subheader("4. Cost-sensitive obesity screening threshold")
    st.caption(
        "Educational sensitivity analysis: Obesity Types I–III are grouped as the current "
        "high-risk label. This is not a clinical diagnosis or a prediction of future disease."
    )
    fn_cost = st.slider(
        "Illustrative cost of one false negative (false-positive cost = 1)",
        min_value=1,
        max_value=10,
        value=5,
        key="detailed_fn_cost",
    )

    if best_model_name in models and hasattr(models[best_model_name], "predict_proba"):
        try:
            _, threshold_X_test, _, threshold_y_test = rebuild_test_split(metadata)
            threshold_pipeline = models[best_model_name]
            risk_levels = ["Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]
            risk_codes = label_encoder.transform(risk_levels)
            classifier_classes = threshold_pipeline.named_steps["classifier"].classes_
            risk_columns = [
                int(np.where(classifier_classes == code)[0][0])
                for code in risk_codes
            ]
            risk_probability = threshold_pipeline.predict_proba(threshold_X_test)[:, risk_columns].sum(axis=1)
            true_risk = np.isin(threshold_y_test, risk_codes).astype(int)

            threshold_rows = []
            for threshold in np.round(np.arange(.10, .91, .05), 2):
                predicted_risk = (risk_probability >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(true_risk, predicted_risk, labels=[0, 1]).ravel()
                threshold_rows.append({
                    "Threshold": threshold,
                    "Sensitivity": tp / (tp + fn) if tp + fn else np.nan,
                    "Specificity": tn / (tn + fp) if tn + fp else np.nan,
                    "Expected Cost": fn_cost * fn + fp,
                    "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                })
            threshold_df = pd.DataFrame(threshold_rows)
            selected = threshold_df.sort_values(["Expected Cost", "FN", "Threshold"]).iloc[0]

            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Recommended threshold", f"{selected['Threshold']:.2f}")
            t2.metric("Sensitivity", f"{selected['Sensitivity']:.1%}")
            t3.metric("Specificity", f"{selected['Specificity']:.1%}")
            t4.metric("Illustrative cost", f"{selected['Expected Cost']:.0f}")

            threshold_long = threshold_df.melt(
                id_vars="Threshold",
                value_vars=["Sensitivity", "Specificity"],
                var_name="Measure",
                value_name="Rate",
            )
            threshold_fig = px.line(
                threshold_long,
                x="Threshold",
                y="Rate",
                color="Measure",
                markers=True,
                title="Sensitivity–specificity trade-off",
            )
            threshold_fig.add_vline(
                x=float(selected["Threshold"]),
                line_dash="dash",
                line_color="#DC2626",
                annotation_text="Lowest cost",
            )
            threshold_fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
            threshold_fig.update_layout(height=480, legend=dict(orientation="h", y=-.18, x=.5, xanchor="center"))
            st.plotly_chart(threshold_fig, use_container_width=True)
            st.markdown(
                f'<div class="insight-card"><b>What it means:</b> Lower thresholds usually catch '
                f'more high-risk-labelled records but create more false positives. With the current '
                f'illustrative {fn_cost}:1 cost ratio, the highlighted threshold minimises '
                f'<code>{fn_cost} × FN + FP</code> on this held-out sample. The threshold must be '
                f'validated externally before any real screening use.</div>',
                unsafe_allow_html=True,
            )
        except Exception as error:
            st.warning(f"Threshold analysis could not be generated: {error}")
    else:
        st.info("The selected model does not expose class probabilities required for threshold analysis.")
