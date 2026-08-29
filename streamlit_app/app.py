"""
BMDS2003 Data Science — Obesity Level Prediction App
======================================================
A Streamlit deployment prototype for the group project
"Estimation of Obesity Levels Based on Eating Habits and Physical Condition".

Run locally with:
    streamlit run app.py

Deploy on Streamlit Community Cloud by pushing this folder
(app.py, requirements.txt, and the models/ folder) to GitHub.
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


MODEL_FILES = {
    "Decision Tree (Baseline)": "decision_tree_pipeline.pkl",
    "Random Forest": "random_forest_pipeline.pkl",
    "SVM": "svm_pipeline.pkl",
    "KNN": "knn_pipeline.pkl",
}


# ============================================================
# OBESITY ORDER
# IMPORTANT:
# This is the order used for:
# - History
# - Confusion Matrix
# - ROC
# - Pie charts
# - Prediction probability chart
# ============================================================

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
# DISPLAY LABELS
# ============================================================

DISPLAY_LABELS = {
    "Insufficient_Weight": "Insufficient Weight",
    "Normal_Weight": "Normal Weight",
    "Overweight_Level_I": "Overweight Level I",
    "Overweight_Level_II": "Overweight Level II",
    "Obesity_Type_I": "Obesity Type I",
    "Obesity_Type_II": "Obesity Type II",
    "Obesity_Type_III": "Obesity Type III",
}


# ============================================================
# COLOURS
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
        "increasing physical activity can help reduce risk of progressing further."
    ),

    "Overweight_Level_II": (
        "Your inputs suggest overweight. A more structured plan combining dietary "
        "adjustment, increased physical activity, and reduced sedentary technology "
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
        "professional promptly to discuss a comprehensive, medically supervised "
        "management plan."
    ),
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def display_level(label):
    """Convert internal class name to readable UI label."""
    return DISPLAY_LABELS.get(
        label,
        str(label).replace("_", " ")
    )


def get_ordinal_indices(label_encoder):
    """
    Convert our desired ordinal class order into the actual
    numeric labels used by the saved LabelEncoder.
    """

    actual_classes = list(label_encoder.classes_)

    indices = []

    for class_name in ORDINAL_ORDER:

        if class_name in actual_classes:
            indices.append(
                actual_classes.index(class_name)
            )

    return indices


def get_ordinal_class_names(label_encoder):
    """
    Return only classes that actually exist in the saved encoder,
    but in the desired ordinal order.
    """

    actual_classes = list(label_encoder.classes_)

    return [
        class_name
        for class_name in ORDINAL_ORDER
        if class_name in actual_classes
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

    label_encoder_path = MODELS_DIR / "label_encoder.pkl"

    if not label_encoder_path.exists():

        raise FileNotFoundError(
            "label_encoder.pkl was not found inside the models folder."
        )

    label_encoder = joblib.load(
        label_encoder_path
    )

    return models, label_encoder


@st.cache_resource(show_spinner=False)
def load_metadata():

    metadata_path = MODELS_DIR / "feature_metadata.json"

    if not metadata_path.exists():

        raise FileNotFoundError(
            "feature_metadata.json was not found inside the models folder."
        )

    with open(
        metadata_path,
        "r",
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
    Recreate the same train/test split used in the notebook.

    IMPORTANT:
    The split must match the original notebook.
    """

    df = load_cleaned_data()

    exclude_cols = [
        "Obesity_Level",
        "BMI",
        "Age_Group"
    ]

    X = df.drop(
        columns=[
            c
            for c in exclude_cols
            if c in df.columns
        ]
    )

    y = df["Obesity_Level"]

    target_classes = _metadata["target_classes"]

    class_to_int = {
        class_name: index
        for index, class_name
        in enumerate(target_classes)
    }

    y_encoded = (
        y.map(class_to_int)
        .values
    )

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

obesity_order = metadata.get(
    "obesity_order",
    ORDINAL_ORDER
)

best_model_name = metadata.get(
    "best_model",
    "Random Forest"
)


# ============================================================
# SESSION STATE — PREDICTION HISTORY
# ============================================================

if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []


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
# FIVE TABS
# ============================================================

(
    tab_about,
    tab_predict,
    tab_explore,
    tab_history,
    tab_performance
) = st.tabs(
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

Applications include:

- **Self-assessment tools** that flag potential risk.
- **Public-health screening** at scale.
- **Targeted lifestyle recommendations**.
- Supporting data-driven obesity research.

**Objective:** classify an individual into one of seven obesity levels
using lifestyle, dietary, and physical-condition attributes.
"""
        )

        st.subheader(
            "Dataset"
        )

        st.markdown(
            """
- **Source:** UCI Machine Learning Repository
- **Records:** 2,111 respondents
- **Features:** 16 input features
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

    with col2:

        st.subheader(
            "Team"
        )

        st.markdown(
            """
| Member | Model |
|---|---|
| Kyra | Decision Tree |
| Liping | Random Forest |
| Wenhsuan | SVM |
| Gladys | KNN |
"""
        )

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
            "the **History** tab to view previous predictions, "
            "and the **Model Performance** tab to evaluate the models."
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
        "a live prediction."
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
        model
        for model in MODEL_FILES
        if model in models
    ]

    if not available_models:

        st.error(
            "No trained models were found."
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
    # RESET STATE
    # ========================================================

    if "prediction_reset_counter" not in st.session_state:

        st.session_state.prediction_reset_counter = 0


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

                type="primary",

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

        # ----------------------------------------------------
        # INPUT DATA
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        pipeline = models[
            chosen_model_name
        ]


        pred_encoded = pipeline.predict(
            input_row
        )[0]


        pred_label = (
            label_encoder
            .inverse_transform(
                [pred_encoded]
            )[0]
        )


        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        history_record = {

            "Time":
                pd.Timestamp.now().strftime(
                    "%H:%M:%S"
                ),

            "Prediction_Number":
                len(
                    st.session_state.prediction_history
                ) + 1,

            "Model":
                chosen_model_name,

            "Obesity_Level":
                pred_label
        }


        st.session_state.prediction_history.append(
            history_record
        )


        # ----------------------------------------------------
        # BMI
        # ----------------------------------------------------

        bmi_value = (
            weight /
            (height ** 2)
        )


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        result_col, chart_col = st.columns(
            [1, 1.3]
        )


        # ====================================================
        # RESULT BOX
        # ====================================================

        with result_col:

            with st.container(
                border=True
            ):

                st.subheader(
                    "🎯 Prediction Result"
                )

                st.metric(

                    "Predicted Obesity Level",

                    display_level(
                        pred_label
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

            with st.container(
                border=True
            ):

                st.subheader(
                    "📊 Class Probabilities"
                )


                if hasattr(
                    pipeline,
                    "predict_proba"
                ):

                    proba = pipeline.predict_proba(
                        input_row
                    )[0]


                    actual_classes = list(
                        label_encoder.classes_
                    )


                    probability_rows = []


                    for class_name in ORDINAL_ORDER:

                        if class_name in actual_classes:

                            class_index = (
                                actual_classes.index(
                                    class_name
                                )
                            )

                            probability_rows.append({

                                "Obesity_Level":
                                    class_name,

                                "Probability":
                                    proba[
                                        class_index
                                    ]

                            })


                    proba_df = pd.DataFrame(
                        probability_rows
                    )


                    proba_df[
                        "Display_Level"
                    ] = proba_df[
                        "Obesity_Level"
                    ].map(
                        display_level
                    )


                    fig = px.bar(

                        proba_df,

                        x="Probability",

                        y="Display_Level",

                        orientation="h",

                        color="Obesity_Level",

                        color_discrete_map=
                            OBESITY_COLORS,

                        category_orders={

                            "Display_Level":
                                [
                                    display_level(x)
                                    for x in ORDINAL_ORDER
                                ]
                        },

                        text="Probability",

                        title=(
                            f"Class Probabilities — "
                            f"{chosen_model_name}"
                        )
                    )


                    fig.update_traces(

                        texttemplate="%{x:.1%}",

                        textposition="outside",

                        hovertemplate=(

                            "<b>Obesity Level:</b> "
                            "%{y}<br>"

                            "<b>Probability:</b> "
                            "%{x:.2%}"

                            "<extra></extra>"
                        )
                    )


                    fig.update_xaxes(

                        range=[
                            0,
                            1
                        ],

                        title="Predicted Probability",

                        tickformat=".0%"
                    )


                    fig.update_yaxes(
                        title="Obesity Level"
                    )


                    fig.update_layout(

                        height=450,

                        showlegend=False,

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


                else:

                    st.info(
                        "This model does not expose "
                        "class probabilities."
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
        f"{df.shape[0]} rows × "
        f"{df.shape[1]} columns"
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

            ORDINAL_ORDER,

            default=ORDINAL_ORDER,

            key="level_filter",

            format_func=display_level
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
                        ORDINAL_ORDER
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
                            ORDINAL_ORDER
                    },

                    hole=0.32
                )


                fig.update_traces(

                    textinfo="percent",

                    textposition="inside",

                    hovertemplate=(

                        "<b>Obesity Level:</b> "
                        "%{label}<br>"

                        "<b>Count:</b> "
                        "%{value}<br>"

                        "<b>Percentage:</b> "
                        "%{percent}"

                        "<extra></extra>"
                    )
                )


                fig.update_layout(

                    height=450,

                    margin=dict(
                        l=0,
                        r=0,
                        t=20,
                        b=100
                    ),

                    legend=dict(

                        title="Obesity Level",

                        orientation="h",

                        x=0.5,

                        xanchor="center",

                        y=-0.25,

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

                        f"<b>{hist_col}</b>: "
                        f"%{{x}}<br>"

                        "<b>Count:</b> "
                        "%{y}"

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
                        ORDINAL_ORDER
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
                    title="Obesity Level"
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
                        ORDINAL_ORDER
                },

                color_discrete_map=
                    OBESITY_COLORS,

                points="outliers",

                title=f"{box_col} by Obesity Level"
            )


            fig.update_traces(

                hovertemplate=(

                    "<b>Obesity Level:</b> "
                    "%{x}<br>"

                    f"<b>{box_col}:</b> "
                    f"%{{y}}"

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

            use_container_width=True
        )


    else:

        st.warning(
            "Choose two different dimensions for rows and columns."
        )


# ============================================================
# TAB 4 — PREDICTION HISTORY
# ============================================================

with tab_history:

    st.header(
        "🕒 Prediction History"
    )

    st.caption(
        "View predictions made during your current Streamlit session."
    )


    # ========================================================
    # NO HISTORY
    # ========================================================

    if not st.session_state.prediction_history:

        st.info(
            "📭 No prediction history yet. "
            "Go to the 🔮 Prediction tab and make a prediction first."
        )


    else:

        # ====================================================
        # HISTORY DATAFRAME
        # ====================================================

        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )


        # ====================================================
        # SUMMARY
        # ====================================================

        total_predictions = len(
            history_df
        )


        most_common_prediction = (

            history_df[
                "Obesity_Level"
            ]

            .value_counts()

            .idxmax()
        )


        most_used_model = (

            history_df[
                "Model"
            ]

            .value_counts()

            .idxmax()
        )


        h1, h2, h3 = st.columns(3)


        with h1:

            st.metric(
                "Total Predictions",
                total_predictions
            )


        with h2:

            st.metric(

                "Most Predicted Level",

                display_level(
                    most_common_prediction
                )
            )


        with h3:

            st.metric(
                "Most Used Model",
                most_used_model
            )


        st.divider()


        # ====================================================
        # HISTORY TABLE
        # ====================================================

        st.subheader(
            "📋 Prediction History"
        )


        display_history = history_df.copy()


        display_history[
            "Obesity_Level"
        ] = (

            display_history[
                "Obesity_Level"
            ]

            .map(
                display_level
            )
        )


        st.dataframe(

            display_history,

            use_container_width=True,

            hide_index=True
        )


        st.divider()


        # ====================================================
        # HISTORY CHARTS
        # ====================================================

        history_chart_col, history_pie_col = st.columns(
            2,
            gap="medium"
        )


        # ====================================================
        # LINE CHART
        # ====================================================

        with history_chart_col:

            with st.container(
                height=650,
                border=True
            ):

                st.markdown(
                    "### 📈 Prediction History"
                )


                line_df = history_df.copy()


                # --------------------------------------------
                # ORDINAL NUMBERS 1–7
                # --------------------------------------------

                ordinal_mapping = {

                    class_name:
                        index + 1

                    for index, class_name
                    in enumerate(
                        ORDINAL_ORDER
                    )
                }


                line_df[
                    "Ordinal"
                ] = (

                    line_df[
                        "Obesity_Level"
                    ]

                    .map(
                        ordinal_mapping
                    )
                )


                line_df[
                    "Display_Level"
                ] = (

                    line_df[
                        "Obesity_Level"
                    ]

                    .map(
                        display_level
                    )
                )


                fig_history = px.line(

                    line_df,

                    x="Prediction_Number",

                    y="Ordinal",

                    markers=True,

                    hover_data=[

                        "Time",
                        "Model",
                        "Display_Level"
                    ],

                    title=(
                        "Obesity Level Prediction Over Time"
                    )
                )


                fig_history.update_traces(

                    hovertemplate=(

                        "<b>Prediction:</b> "
                        "%{x}<br>"

                        "<b>Obesity Level:</b> "
                        "%{customdata[2]}<br>"

                        "<b>Model:</b> "
                        "%{customdata[1]}<br>"

                        "<b>Time:</b> "
                        "%{customdata[0]}"

                        "<extra></extra>"
                    )
                )


                # --------------------------------------------
                # Y-AXIS = 1 TO 7
                # --------------------------------------------

                fig_history.update_yaxes(

                    tickmode="array",

                    tickvals=list(
                        range(
                            1,
                            len(
                                ORDINAL_ORDER
                            ) + 1
                        )
                    ),

                    ticktext=[

                        display_level(
                            x
                        )

                        for x
                        in ORDINAL_ORDER
                    ],

                    title="Obesity Level",

                    range=[

                        0.5,

                        len(
                            ORDINAL_ORDER
                        ) + 0.5
                    ]
                )


                fig_history.update_xaxes(

                    title="Prediction Number",

                    dtick=1
                )


                fig_history.update_layout(

                    height=500,

                    margin=dict(
                        l=30,
                        r=20,
                        t=70,
                        b=50
                    )
                )


                st.plotly_chart(

                    fig_history,

                    use_container_width=True,

                    config={
                        "displayModeBar": False
                    }
                )


        # ====================================================
        # PIE CHART
        # ====================================================

        with history_pie_col:

            with st.container(
                height=650,
                border=True
            ):

                st.markdown(
                    "### 🥧 Prediction Distribution"
                )


                history_counts = (

                    history_df[
                        "Obesity_Level"
                    ]

                    .value_counts()

                    .reindex(
                        ORDINAL_ORDER
                    )

                    .fillna(0)

                    .reset_index()
                )


                history_counts.columns = [
                    "Obesity_Level",
                    "Count"
                ]


                history_counts = history_counts[
                    history_counts["Count"] > 0
                ]


                fig_history_pie = px.pie(

                    history_counts,

                    names="Obesity_Level",

                    values="Count",

                    color="Obesity_Level",

                    color_discrete_map=
                        OBESITY_COLORS,

                    category_orders={

                        "Obesity_Level":
                            ORDINAL_ORDER
                    },

                    hole=0.32,

                    title=(
                        "Predicted Obesity Level Distribution"
                    )
                )


                fig_history_pie.update_traces(

                    textinfo="percent",

                    textposition="inside",

                    hovertemplate=(

                        "<b>Obesity Level:</b> "
                        "%{label}<br>"

                        "<b>Predictions:</b> "
                        "%{value}<br>"

                        "<b>Percentage:</b> "
                        "%{percent}"

                        "<extra></extra>"
                    )
                )


                fig_history_pie.update_layout(

                    height=500,

                    margin=dict(
                        l=10,
                        r=10,
                        t=70,
                        b=120
                    ),

                    legend=dict(

                        title="Obesity Level",

                        orientation="h",

                        x=0.5,

                        xanchor="center",

                        y=-0.25,

                        yanchor="top",

                        font=dict(
                            size=9
                        )
                    )
                )


                st.plotly_chart(

                    fig_history_pie,

                    use_container_width=True,

                    config={
                        "displayModeBar": False
                    }
                )


        # ====================================================
        # CLEAR HISTORY
        # ====================================================

        st.divider()


        clear_col1, clear_col2, clear_col3 = st.columns(
            [1, 1, 1]
        )


        with clear_col2:

            if st.button(

                "🗑️ Clear Prediction History",

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

        "`Train_Accuracy_InSample` is the model scored on the "
        "exact rows it was fit on. `Train_Accuracy_CV` is based "
        "on held-out training folds and is more meaningful for "
        "checking generalisation."
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

            range=[
                0,
                1.05
            ],

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
    # 3. TRAIN VS TEST
    # ========================================================

    st.subheader(
        "Inner Comparison — Train vs Test Accuracy"
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

                title=(
                    "Cross-Validated Train vs Test Accuracy"
                )
            )


            fig.update_traces(

                texttemplate="%{text:.1%}",

                textposition="outside",

                hovertemplate=(

                    "<b>Model:</b> %{x}<br>"

                    "<b>Dataset:</b> "
                    "%{fullData.name}<br>"

                    "<b>Accuracy:</b> "
                    "%{y:.2%}"

                    "<extra></extra>"
                )
            )


            fig.update_yaxes(

                range=[
                    0,
                    1.05
                ],

                title="Accuracy",

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

                "Cross-validated train accuracy is more meaningful "
                "for checking generalisation because each validation "
                "fold contains observations not used to fit that fold."
            )


    # ========================================================
    # 3B. IN SAMPLE
    # ========================================================

    with inner_tab2:

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

                    "<b>Dataset:</b> "
                    "%{fullData.name}<br>"

                    "<b>Accuracy:</b> "
                    "%{y:.2%}"

                    "<extra></extra>"
                )
            )


            fig.update_yaxes(

                range=[
                    0,
                    1.05
                ],

                title="Accuracy",

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

                "In-sample accuracy evaluates the model on the "
                "same training observations used during fitting."
            )


    # ========================================================
    # 4. WEIGHTED VS MACRO
    # ========================================================

    st.subheader(
        "Weighted vs Macro-averaged Metrics"
    )


    st.caption(

        "Weighted averages account for class frequency, while "
        "macro averages give every class equal importance."
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

                metric_long[
                    "Average"
                ]

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

                title=(
                    f"{metric}: Weighted vs Macro"
                )
            )


            fig.update_traces(

                texttemplate="%{text:.1%}",

                textposition="outside",

                hovertemplate=(

                    "<b>Model:</b> %{x}<br>"

                    "<b>Average:</b> "
                    "%{fullData.name}<br>"

                    f"<b>{metric}:</b> "
                    "%{y:.2%}"

                    "<extra></extra>"
                )
            )


            fig.update_yaxes(

                range=[
                    0,
                    1.05
                ],

                title=metric,

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
    # 5. MODEL EVALUATION
    # ========================================================

    st.divider()


    st.subheader(
        "Model Evaluation"
    )


    st.caption(

        "Select a model to inspect its test-set predictions, "
        "confusion matrix, ROC curves and classification report."
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

            if best_model_name
            in available_eval_models

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


        # ====================================================
        # PREDICTION
        # ====================================================

        y_pred = eval_pipeline.predict(
            X_test
        )


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


            with m1:

                st.metric(

                    "Test Accuracy",

                    f"{selected_row['Test_Accuracy']:.1%}"
                )


            with m2:

                st.metric(

                    "Precision",

                    f"{selected_row['Precision_Weighted']:.1%}"
                )


            with m3:

                st.metric(

                    "Recall",

                    f"{selected_row['Recall_Weighted']:.1%}"
                )


            with m4:

                st.metric(

                    "F1",

                    f"{selected_row['F1_Weighted']:.1%}"
                )


            st.markdown(
                "**Macro-average metrics**"
            )


            m5, m6, m7, m8 = st.columns(4)


            with m5:

                st.metric(
                    " ",
                    " "
                )


            with m6:

                st.metric(

                    "Precision",

                    f"{selected_row['Precision_Macro']:.1%}"
                )


            with m7:

                st.metric(

                    "Recall",

                    f"{selected_row['Recall_Macro']:.1%}"
                )


            with m8:

                st.metric(

                    "F1",

                    f"{selected_row['F1_Macro']:.1%}"
                )


        else:

            st.warning(
                f"'{eval_model_name}' was not found in the comparison table."
            )


        # ====================================================
        # LARGE BOX FOR CM + ROC
        # ====================================================

        st.markdown("### 📊 Classification Performance")


        with st.container(
            border=True
        ):

            st.markdown(
                "#### Confusion Matrix & ROC Analysis"
            )


            cm_col, roc_col = st.columns(
                2,
                gap="medium"
            )


            # ====================================================
            # CONFUSION MATRIX
            # ====================================================

            with cm_col:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        "##### Confusion Matrix"
                    )


                    # --------------------------------------------
                    # ACTUAL LABEL ENCODER CLASSES
                    # --------------------------------------------

                    actual_classes = list(
                        label_encoder.classes_
                    )


                    # --------------------------------------------
                    # GET CLASSES IN ORDINAL ORDER
                    # --------------------------------------------

                    ordinal_classes = [

                        class_name

                        for class_name
                        in ORDINAL_ORDER

                        if class_name
                        in actual_classes
                    ]


                    # --------------------------------------------
                    # CONVERT TO NUMERIC LABELS
                    # --------------------------------------------

                    ordinal_indices = [

                        actual_classes.index(
                            class_name
                        )

                        for class_name
                        in ordinal_classes
                    ]


                    # --------------------------------------------
                    # SAFETY CHECK
                    # --------------------------------------------

                    if len(
                        ordinal_indices
                    ) == 0:

                        st.error(

                            "No matching obesity classes were "
                            "found in the saved label encoder."
                        )

                    else:

                        # ----------------------------------------
                        # CONFUSION MATRIX
                        # ----------------------------------------

                        cm = confusion_matrix(

                            y_test,

                            y_pred,

                            labels=ordinal_indices
                        )


                        # ----------------------------------------
                        # DATAFRAME
                        # ----------------------------------------

                        cm_display_names = [

                            display_level(
                                class_name
                            )

                            for class_name
                            in ordinal_classes
                        ]


                        cm_df = pd.DataFrame(

                            cm,

                            index=cm_display_names,

                            columns=cm_display_names
                        )


                        # ----------------------------------------
                        # PLOT
                        # ----------------------------------------

                        fig_cm = px.imshow(

                            cm_df,

                            text_auto=True,

                            aspect="auto",

                            labels={

                                "x": "Predicted",

                                "y": "Actual",

                                "color": "Count"
                            },

                            title=(
                                f"Confusion Matrix — "
                                f"{eval_model_name}"
                            )
                        )


                        fig_cm.update_traces(

                            hovertemplate=(

                                "<b>Actual:</b> "
                                "%{y}<br>"

                                "<b>Predicted:</b> "
                                "%{x}<br>"

                                "<b>Count:</b> "
                                "%{z}"

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

                                b=150
                            )
                        )


                        st.plotly_chart(

                            fig_cm,

                            use_container_width=True,

                            config={

                                "displayModeBar":
                                    False
                            }
                        )


            # ====================================================
            # ROC CURVE
            # ====================================================

            with roc_col:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        "##### Per-class ROC Curves"
                    )


                    if hasattr(

                        eval_pipeline,

                        "predict_proba"
                    ):

                        # ----------------------------------------
                        # LABEL ENCODER CLASSES
                        # ----------------------------------------

                        actual_classes = list(
                            label_encoder.classes_
                        )


                        # ----------------------------------------
                        # ORDINAL CLASSES
                        # ----------------------------------------

                        ordinal_classes = [

                            class_name

                            for class_name
                            in ORDINAL_ORDER

                            if class_name
                            in actual_classes
                        ]


                        # ----------------------------------------
                        # NUMERIC INDICES
                        # ----------------------------------------

                        ordinal_indices = [

                            actual_classes.index(
                                class_name
                            )

                            for class_name
                            in ordinal_classes
                        ]


                        # ----------------------------------------
                        # PREDICT PROBABILITIES
                        # ----------------------------------------

                        y_proba = (
                            eval_pipeline
                            .predict_proba(
                                X_test
                            )
                        )


                        # ----------------------------------------
                        # ROC DATA
                        # ----------------------------------------

                        roc_rows = []


                        for class_name, class_index in zip(

                            ordinal_classes,

                            ordinal_indices
                        ):

                            # ------------------------------------
                            # BINARY TARGET
                            # ------------------------------------

                            y_true_binary = (

                                y_test
                                == class_index

                            ).astype(int)


                            # ------------------------------------
                            # CHECK BOTH CLASSES EXIST
                            # ------------------------------------

                            if len(
                                np.unique(
                                    y_true_binary
                                )
                            ) < 2:

                                continue


                            # ------------------------------------
                            # PROBABILITY
                            # ------------------------------------

                            y_score = (

                                y_proba[
                                    :,
                                    class_index
                                ]
                            )


                            # ------------------------------------
                            # ROC
                            # ------------------------------------

                            fpr, tpr, _ = roc_curve(

                                y_true_binary,

                                y_score
                            )


                            # ------------------------------------
                            # AUC
                            # ------------------------------------

                            roc_auc_value = auc(

                                fpr,

                                tpr
                            )


                            # ------------------------------------
                            # STORE POINTS
                            # ------------------------------------

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
                                        display_level(
                                            class_name
                                        ),

                                    "AUC":
                                        roc_auc_value
                                })


                        # ----------------------------------------
                        # CHECK ROC DATA
                        # ----------------------------------------

                        if not roc_rows:

                            st.warning(
                                "ROC curves could not be calculated "
                                "for the selected model."
                            )

                        else:

                            roc_df = pd.DataFrame(
                                roc_rows
                            )


                            # ------------------------------------
                            # ROC PLOT
                            # ------------------------------------

                            fig_roc = px.line(

                                roc_df,

                                x=(
                                    "False Positive Rate"
                                ),

                                y=(
                                    "True Positive Rate"
                                ),

                                color="Obesity Level",

                                title=(
                                    f"ROC Curves — "
                                    f"{eval_model_name}"
                                ),

                                hover_data=[
                                    "AUC"
                                ]
                            )


                            # ------------------------------------
                            # RANDOM CLASSIFIER
                            # ------------------------------------

                            fig_roc.add_trace(

                                go.Scatter(

                                    x=[
                                        0,
                                        1
                                    ],

                                    y=[
                                        0,
                                        1
                                    ],

                                    mode="lines",

                                    name="Random Classifier",

                                    line=dict(
                                        dash="dot"
                                    )
                                )
                            )


                            # ------------------------------------
                            # AXES
                            # ------------------------------------

                            fig_roc.update_xaxes(

                                range=[
                                    0,
                                    1
                                ],

                                title="False Positive Rate"
                            )


                            fig_roc.update_yaxes(

                                range=[
                                    0,
                                    1
                                ],

                                title="True Positive Rate"
                            )


                            # ------------------------------------
                            # LAYOUT
                            # ------------------------------------

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

                                use_container_width=True,

                                config={

                                    "displayModeBar":
                                        False
                                }
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

                labels=list(
                    range(
                        len(
                            label_encoder.classes_
                        )
                    )
                ),

                target_names=[

                    display_level(
                        x
                    )

                    for x
                    in label_encoder.classes_
                ],

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

        eval_model_name
        == "Random Forest"
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

                    "<b>Importance:</b> "
                    "%{x:.3f}"

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
