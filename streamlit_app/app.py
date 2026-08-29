"""
BMDS2003 Data Science — Obesity Level Prediction App
======================================================
A Streamlit deployment prototype for the group project
"Estimation of Obesity Levels Based on Eating Habits and Physical Condition".

Run locally with:
    streamlit run app.py

Deploy on Streamlit Community Cloud by pushing this folder (app.py,
requirements.txt, and the models/ folder) to a GitHub repository and
pointing Streamlit Cloud at app.py.
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
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

# ============================================================
# PAGE CONFIG (must be the first Streamlit call)
# ============================================================

st.set_page_config(
    page_title="Obesity Level Predictor | BMDS2003",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
# CACHED LOADERS
# ============================================================

@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    models = {}
    for name, filename in MODEL_FILES.items():
        path = MODELS_DIR / filename
        if path.exists():
            models[name] = joblib.load(path)
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
    return models, label_encoder


@st.cache_resource(show_spinner=False)
def load_metadata():
    with open(MODELS_DIR / "feature_metadata.json") as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading dataset...")
def load_cleaned_data():
    return pd.read_csv(MODELS_DIR / "obesity_cleaned.csv")


@st.cache_data(show_spinner=False)
def load_comparison_table():
    return pd.read_csv(MODELS_DIR / "model_comparison.csv", index_col=0)


@st.cache_data(show_spinner="Rebuilding the held-out test split for evaluation...")
def rebuild_test_split(_metadata):
    """Recreate the EXACT same train/test split used in the notebook (same
    random_state, test_size, and stratification) so that live evaluation in
    the Model Performance tab matches the notebook's reported numbers."""
    df = load_cleaned_data()
    exclude_cols = ["Obesity_Level", "BMI", "Age_Group"]
    X = df.drop(columns=[c for c in exclude_cols if c in df.columns])
    y = df["Obesity_Level"]

    target_classes = _metadata["target_classes"]
    class_to_int = {c: i for i, c in enumerate(target_classes)}
    y_encoded = y.map(class_to_int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    return X_train, X_test, y_train, y_test


# ============================================================
# LOAD EVERYTHING ONCE
# ============================================================

models, label_encoder = load_models()
metadata = load_metadata()
obesity_order = metadata["obesity_order"]
best_model_name = metadata.get("best_model", "Random Forest")

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
# HEADER
# ============================================================

st.title("🍎 Obesity Level Prediction System")
st.caption(
    "BMDS2003 Data Science Group Project — Estimation of Obesity Levels Based on "
    "Eating Habits and Physical Condition (CRISP-DM)"
)

tab_about, tab_predict, tab_explore, tab_performance = st.tabs(
    ["🏠 About", "🔮 Prediction", "📊 Data Exploration", "📈 Model Performance"]
)

# ============================================================
# TAB 1 — ABOUT
# ============================================================
with tab_about:
    st.header("About This Project")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Business Understanding")
        st.markdown(
            """
Obesity is a growing public-health concern linked to diabetes, cardiovascular disease, and
reduced quality of life. Traditional diagnosis relies on clinical BMI measurement, which
requires an in-person visit. This project explores whether **everyday lifestyle and eating
habits alone can predict a person's obesity category**, enabling:

- **Self-assessment tools** that flag risk before a clinical visit is needed.
- **Public-health screening** at scale (schools, workplaces, community programmes).
- **Targeted lifestyle recommendations** that highlight the *actionable* factors (diet,
  activity, transport habits) most associated with each obesity category.

**Objective:** classify an individual into one of seven clinically-defined obesity levels using
16 lifestyle, dietary, and physical-condition attributes.
            """
        )

        st.subheader("Dataset")
        st.markdown(
            """
- **Source:** UCI Machine Learning Repository — *Estimation of Obesity Levels Based on Eating
  Habits and Physical Condition*.
- **Records:** 2,111 respondents (Mexico, Peru, Colombia), 17 columns (16 features + target).
- **Target:** `Obesity_Level` — 7 classes ranging from `Insufficient_Weight` to
  `Obesity_Type_III`.
            """
        )

        st.subheader("CRISP-DM Workflow")
        st.markdown(
            """
1. **Business Understanding** — define the prediction problem and its real-world value.
2. **Data Understanding** — profile the dataset's structure, types, and distributions.
3. **Data Preparation** — clean, rename, check outliers (IQR / Z-score / Modified Z-score),
   engineer BMI (for analysis only), and demonstrate standardisation & binning techniques.
4. **Modelling** — train 4 classifiers: a Decision Tree baseline, plus hyperparameter-tuned
   Random Forest, SVM, and KNN models, each validated with 5-fold cross-validation.
5. **Evaluation** — compare models with accuracy, precision, recall, F1, AUC, confusion
   matrices, and an explicit train-vs-test *overfitting* check.
6. **Deployment** — this Streamlit application, which lets users generate live predictions,
   explore the data, and inspect model performance.
            """
        )

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
        st.success(f"**{best_model_name}** achieved the highest test-set accuracy.")

        comparison_preview = load_comparison_table()
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

    available_models = [m for m in MODEL_FILES if m in models]
    default_index = (
        available_models.index(best_model_name) if best_model_name in available_models else 0
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
    st.caption(f"Using **{chosen_model_name}**" + (" (best on test set)" if chosen_model_name == best_model_name else ""))

    num_meta = metadata["numeric_features"]
    cat_meta = metadata["categorical_features"]

    # Logical (non-alphabetical) ordering for the two ordinal frequency fields
    FREQUENCY_ORDER = ["no", "Sometimes", "Frequently", "Always"]
    freq_label = lambda x: "Never" if x == "no" else x  # noqa: E731
    yes_no_label = lambda x: "✅ Yes" if x == "yes" else "❌ No"  # noqa: E731

    with st.form("prediction_form"):
        st.subheader("Personal & Physical Attributes")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Gender**")
            gender = st.radio(
                "Gender", cat_meta["Gender"], horizontal=True, label_visibility="collapsed",
                format_func=lambda x: "👩 Female" if x == "Female" else "👨 Male",
            )
            age = st.number_input(
                "Age (years)",
                min_value=float(num_meta["Age"]["min"]),
                max_value=100.0,
                value=round(num_meta["Age"]["mean"], 1),
                step=1.0,
            )
        with c2:
            height = st.number_input(
                "Height (m)",
                min_value=float(num_meta["Height"]["min"]),
                max_value=float(num_meta["Height"]["max"]) + 0.3,
                value=round(num_meta["Height"]["mean"], 2),
                step=0.01,
                format="%.2f",
            )
        with c3:
            weight = st.number_input(
                "Weight (kg)",
                min_value=float(num_meta["Weight"]["min"]),
                max_value=float(num_meta["Weight"]["max"]) + 50.0,
                value=round(num_meta["Weight"]["mean"], 1),
                step=1.0,
            )

        st.subheader("Eating Habits")
        c4, c5, c6 = st.columns(3)
        with c4:
            st.markdown("**Family history of overweight?**")
            family_history = st.radio(
                "Family history of overweight?", cat_meta["Family_History_Overweight"],
                horizontal=True, label_visibility="collapsed", format_func=yes_no_label,
            )
            st.markdown("**Frequently eats high-caloric food?**")
            favc = st.radio(
                "Frequently eats high-caloric food?", cat_meta["Frequent_High_Caloric_Food"],
                horizontal=True, label_visibility="collapsed", format_func=yes_no_label,
            )
        with c5:
            fcvc = st.slider(
                "Vegetable consumption frequency (1 = never, 3 = always)",
                1.0, 3.0, round(num_meta["Vegetable_Consumption_Freq"]["mean"], 1), 0.1,
            )
            ncp = st.slider(
                "Number of main meals per day",
                1.0, 4.0, round(num_meta["Main_Meals_Per_Day"]["mean"], 1), 0.5,
            )
        with c6:
            st.markdown("**Eats food between meals?**")
            caec = st.select_slider(
                "Eats food between meals?", options=FREQUENCY_ORDER, value="Sometimes",
                label_visibility="collapsed", format_func=freq_label,
            )
            st.markdown("**Alcohol consumption**")
            calc = st.select_slider(
                "Alcohol consumption", options=FREQUENCY_ORDER, value="Sometimes",
                label_visibility="collapsed", format_func=freq_label,
            )

        st.subheader("Lifestyle & Physical Condition")
        c7, c8, c9 = st.columns(3)
        with c7:
            st.markdown("**Smokes?**")
            smoke = st.radio(
                "Smokes?", cat_meta["Smokes"], horizontal=True,
                label_visibility="collapsed", format_func=yes_no_label,
            )
            st.markdown("**Monitors calorie intake?**")
            scc = st.radio(
                "Monitors calorie intake?", cat_meta["Calorie_Monitoring"], horizontal=True,
                label_visibility="collapsed", format_func=yes_no_label,
            )
        with c8:
            ch2o = st.slider(
                "Daily water intake (1 = <1L, 3 = >2L)",
                1.0, 3.0, round(num_meta["Daily_Water_Intake"]["mean"], 1), 0.1,
            )
            faf = st.slider(
                "Physical activity frequency (0 = none, 3 = frequent)",
                0.0, 3.0, round(num_meta["Physical_Activity_Freq"]["mean"], 1), 0.1,
            )
        with c9:
            tue = st.slider(
                "Technology usage time (0 = low, 2 = high)",
                0.0, 2.0, round(num_meta["Technology_Usage_Time"]["mean"], 1), 0.1,
            )
            st.markdown("**Usual transportation mode**")
            mtrans = st.pills(
                "Usual transportation mode", cat_meta["Transportation_Mode"],
                default=cat_meta["Transportation_Mode"][0], label_visibility="collapsed",
                format_func=lambda x: x.replace("_", " "),
            )
            if mtrans is None:
                mtrans = cat_meta["Transportation_Mode"][0]

        submitted = st.form_submit_button("Predict Obesity Level", type="primary")

    if submitted:
        input_row = pd.DataFrame([{
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
        }])

        pipeline = models[chosen_model_name]
        pred_encoded = pipeline.predict(input_row)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        bmi_value = weight / (height ** 2)

        st.divider()
        result_col, chart_col = st.columns([1, 1.3])

        with result_col:
            st.subheader("Prediction Result")
            st.metric("Predicted Obesity Level", pred_label.replace("_", " "))
            st.metric("Computed BMI", f"{bmi_value:.1f} kg/m²")
            st.caption(f"Model used: **{chosen_model_name}**")
            st.markdown("**Recommendation:**")
            st.write(RECOMMENDATIONS.get(pred_label, "Consult a healthcare professional for guidance."))

        with chart_col:
            if hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba(input_row)[0]
                proba_df = pd.DataFrame({
                    "Obesity_Level": label_encoder.classes_,
                    "Probability": proba,
                }).set_index("Obesity_Level").reindex(obesity_order)

                fig = px.bar(
                    proba_df.reset_index(),
                    x="Probability",
                    y="Obesity_Level",
                    orientation="h",
                    title=f"Class Probabilities — {chosen_model_name}",
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
                    title="Predicted Probability"
                )

                fig.update_yaxes(
                    title="Obesity Level"
                )

                fig.update_layout(
                    height=450
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )
            else:
                st.info("This model does not expose class probabilities.")

# ============================================================
# TAB 3 — DATA EXPLORATION
# ============================================================
with tab_explore:

    st.header("📊 Data Exploration")

    df = load_cleaned_data()

    st.caption(
        f"Cleaned dataset: {df.shape[0]} rows × {df.shape[1]} columns"
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

        st.write("Numeric summary:")

        st.dataframe(
            df.describe().T,
            width="stretch"
        )

    # ========================================================
    # FILTERS
    # ========================================================

    st.subheader("🔎 Filters")

    fc1, fc2, fc3 = st.columns(3)

    with fc1:

        gender_filter = st.multiselect(
            "Gender",
            sorted(df["Gender"].unique()),
            default=sorted(df["Gender"].unique()),
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
        df["Gender"].isin(gender_filter)
        & df["Obesity_Level"].isin(level_filter)
        & df["Age"].between(*age_range)
    ]

    st.caption(
        f"Showing {len(filtered)} of {len(df)} records after filtering."
    )

    # ========================================================
    # NUMERIC OPTIONS
    # ========================================================

    num_options = filtered.select_dtypes(
        include=np.number
    ).columns.tolist()

    # ========================================================
    # 1. PIE CHART
    # ========================================================

    chart1, chart2 = st.columns(2)

    with chart1:

        st.markdown("### 🍩 Obesity Level Distribution")

        counts = (
            filtered["Obesity_Level"]
            .value_counts()
            .reindex(obesity_order)
            .fillna(0)
            .reset_index()
        )

        counts.columns = [
            "Obesity_Level",
            "Count"
        ]

        # Remove zero-count categories
        counts = counts[counts["Count"] > 0]

        fig = px.pie(
            counts,
            names="Obesity_Level",
            values="Count",
            title="Obesity Level Distribution",
            color="Obesity_Level",
            color_discrete_map=OBESITY_COLORS,
            category_orders={
                "Obesity_Level": obesity_order
            },
            hole=0.35
        )

        fig.update_traces(
            textinfo="percent",
            textposition="inside",
            hovertemplate=(
                "<b>Obesity Level:</b> %{label}<br>"
                "<b>Count:</b> %{value}<br>"
                "<b>Percentage:</b> %{percent}"
                "<extra></extra>"
            )
        )

        fig.update_layout(
            height=500,
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
# 2. CLEAR HISTOGRAM
# ========================================================
    with chart2:

    st.markdown("### 📊 Numeric Distribution")

    hist_col = st.selectbox(
        "Choose a numeric variable",
        num_options,
        index=(
            num_options.index("BMI")
            if "BMI" in num_options
            else 0
        ),
        key="hist_col"
    )

    # Number of bins
    hist_bins = st.slider(
        "Number of bins",
        min_value=5,
        max_value=30,
        value=15,
        step=5,
        key="hist_bins"
    )

    # ----------------------------------------------------
    # HISTOGRAM — SEPARATE PANELS
    # ----------------------------------------------------

    fig = px.histogram(
        filtered,
        x=hist_col,
        facet_col="Obesity_Level",
        facet_col_wrap=2,
        nbins=hist_bins,

        category_orders={
            "Obesity_Level": obesity_order
        },

        color="Obesity_Level",
        color_discrete_map=OBESITY_COLORS,

        title=f"Distribution of {hist_col} by Obesity Level"
    )

    # ----------------------------------------------------
    # CLEAN UP
    # ----------------------------------------------------

    fig.update_traces(
        marker_line_width=0,
        hovertemplate=(
            f"<b>{hist_col}:</b> %{{x}}<br>"
            "<b>Count:</b> %{y}"
            "<extra></extra>"
        )
    )

    # Remove repeated facet labels
    fig.for_each_annotation(
        lambda a: a.update(
            text=a.text.split("=")[-1]
        )
    )

    # Make all panels use the same x-axis range
    fig.update_xaxes(
        matches="x",
        showgrid=False,
        title_text=hist_col
    )

    fig.update_yaxes(
        showgrid=True,
        title_text="Number of Records"
    )

    fig.update_layout(
        height=850,
        bargap=0.08,

        # Hide legend because each panel already has its title
        showlegend=False,

        margin=dict(
            l=60,
            r=30,
            t=80,
            b=60
        ),

        hovermode="closest"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "💡 Each panel represents one obesity level, making it easier "
        "to compare the distribution of the selected numeric variable "
        "without overlapping categories."
    )

    # ========================================================
    # 3. SCATTERPLOT
    # ========================================================

    st.divider()

    chart3, chart4 = st.columns(2)

    with chart3:

        st.markdown("### 🔵 Interactive Scatterplot")

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
                else min(1, len(num_options) - 1)
            ),
            key="scatter_y"
        )

        fig = px.scatter(
            filtered,
            x=sx,
            y=sy,
            color="Obesity_Level",
            category_orders={
                "Obesity_Level": obesity_order
            },
            color_discrete_map=OBESITY_COLORS,
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
    # 4. BOXPLOT
    # ========================================================

    with chart4:

        st.markdown("### 📦 Interactive Boxplot")

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
                "Obesity_Level": obesity_order
            },
            color_discrete_map=OBESITY_COLORS,
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
    # 5. CORRELATION HEATMAP
    # ========================================================

    st.divider()

    st.subheader("🔥 Interactive Correlation Heatmap")

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
    # 6. OLAP PIVOT EXPLORER
    # ========================================================

    st.divider()

    st.subheader(
        "📋 OLAP-style Pivot Explorer"
    )

    st.caption(
        "Build your own multidimensional summary table by choosing "
        "row and column dimensions, a numeric measure, and an aggregation."
    )

    categorical_options = filtered.select_dtypes(
        exclude=np.number
    ).columns.tolist()

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
            key="row_dimension"
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
            width="stretch"
        )

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

    st.subheader("Consolidated Comparison Table")
    st.caption(
        "`Train_Accuracy_InSample` is the model scored on the exact rows it was fit on — it is "
        "expected to sit near/at 1.0 for flexible models (Decision Tree, KNN) and is **not** a "
        "reliable overfitting signal by itself. `Train_Accuracy_CV` (cross-validated on held-out "
        "training folds) is the one to trust for the overfitting check below."
    )
    st.dataframe(comparison_df.style.format(precision=4), width='stretch')

    st.subheader("Outer Comparison — Models Against Each Other (Test Set)")
    outer_metrics = ["Test_Accuracy", "Precision_Weighted", "Recall_Weighted", "F1_Weighted"]
    outer_plot_df = (
    comparison_df[outer_metrics]
    .reset_index()
    .rename(columns={"index": "Model"})
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
        title="Score"
    )

    fig.update_layout(
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Inner Comparison — Train vs Test Accuracy (Overfitting Check)")
    inner_tab1, inner_tab2 = st.tabs(["Cross-validated (trust this)", "In-sample (for transparency)"])

    with inner_tab1:
        st.subheader("Inner Comparison — Train vs Test Accuracy")

        accuracy_df = (
            comparison_df[["Train_Accuracy_CV", "Test_Accuracy"]]
            .reset_index()
            .rename(columns={"index": "Model"})
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
            title="Train vs Test Accuracy"
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
            "Each fold's model is scored on held-out rows of the training data, so this number "
            "cannot trivially reach 1.0 through memorisation — this is the meaningful overfitting "
            "check. All four models here show a small gap (a few percentage points either way), "
            "indicating **no serious overfitting** once measured correctly."
        )

    with inner_tab2:
        accuracy_df = (
            comparison_df[["Train_Accuracy_CV", "Test_Accuracy"]]
            .reset_index()
            .rename(columns={"index": "Model"})
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
            title="Train vs Test Accuracy"
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
            "In-sample accuracy = fit and predict on the SAME training rows. It measures "
            "memorisation, not generalisation, and is expected to be very high — exactly 1.0 for "
            "the Decision Tree (unconstrained by default) and KNN (weights='distance', so every "
            "training point's nearest neighbour during scoring is itself, at distance 0). It is "
            "shown here for transparency, not as an overfitting diagnostic."
        )

    st.subheader("Weighted vs Macro-averaged Metrics")
    st.caption(
        "Weighted Recall is mathematically identical to Accuracy for any single-label multiclass "
        "problem (not a computation quirk). Macro metrics weight every class equally regardless "
        "of size and diverge more visibly whenever a model is weaker on specific classes — see "
        "how much further KNN's macro scores drop below its weighted scores below."
    )
    macro_metric_tabs = st.tabs(["Precision", "Recall", "F1"])
    for tab, metric in zip(macro_metric_tabs, ["Precision", "Recall", "F1"]):
        with tab:
            metric_df = (
                comparison_df[
                    [f"{metric}_Weighted", f"{metric}_Macro"]
                ]
                .reset_index()
                .rename(columns={"index": "Model"})
            )

            metric_long = metric_df.melt(
                id_vars="Model",
                var_name="Average",
                value_name="Score"
            )
            
            metric_long["Average"] = metric_long["Average"].str.replace(
                f"{metric}_", "", regex=False
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
            
            fig.update_layout(
                height=450
            )
            
            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.divider()
    st.subheader("Live Evaluation for a Selected Model")
    st.caption(
        "Recomputed live on the same held-out test split used in the notebook "
        "(identical random_state and stratification), so results match exactly."
    )

    eval_model_name = st.selectbox(
        "Choose a model to inspect", list(models.keys()),
        index=list(models.keys()).index(best_model_name) if best_model_name in models else 0,
        key="eval_model",
    )

    X_train, X_test, y_train, y_test = rebuild_test_split(metadata)
    eval_pipeline = models[eval_model_name]
    y_pred = eval_pipeline.predict(X_test)

    st.markdown("**Weighted-average metrics**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
    m2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.1%}")
    m3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.1%}")
    m4.metric("F1", f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.1%}")

    st.markdown("**Macro-average metrics** (every class counted equally)")
    m5, m6, m7, m8 = st.columns(4)
    m5.metric("—", "")
    m6.metric("Precision", f"{precision_score(y_test, y_pred, average='macro', zero_division=0):.1%}")
    m7.metric("Recall", f"{recall_score(y_test, y_pred, average='macro', zero_division=0):.1%}")
    m8.metric("F1", f"{f1_score(y_test, y_pred, average='macro', zero_division=0):.1%}")

    cm_col, roc_col = st.columns(2)

    with cm_col:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)

        cm_df = pd.DataFrame(
            cm,
            index=label_encoder.classes_,
            columns=label_encoder.classes_
        )
        
        fig = px.imshow(
            cm_df,
            text_auto=True,
            aspect="auto",
            title=f"Confusion Matrix — {eval_model_name}",
            labels={
                "x": "Predicted",
                "y": "Actual",
                "color": "Count"
            }
        )
        
        fig.update_traces(
            hovertemplate=(
                "<b>Actual:</b> %{y}<br>"
                "<b>Predicted:</b> %{x}<br>"
                "<b>Count:</b> %{z}"
                "<extra></extra>"
            )
        )
        
        fig.update_layout(
            height=550
        )
        
        st.plotly_chart(
            fig,
            use_container_width=True
        )

        with roc_col:
            st.markdown("**Per-class ROC Curves**")

            if hasattr(eval_pipeline, "predict_proba"):
                y_proba = eval_pipeline.predict_proba(X_test)
                y_test_bin = label_binarize(
                    y_test,
                    classes=list(range(len(label_encoder.classes_)))
                )

                roc_rows = []
    
                for i, class_name in enumerate(label_encoder.classes_):
    
                    fpr, tpr, _ = roc_curve(
                        y_test_bin[:, i],
                        y_proba[:, i]
                    )
    
                    roc_auc_i = auc(fpr, tpr)
    
                    for x, y in zip(fpr, tpr):
                        roc_rows.append({
                            "False Positive Rate": x,
                            "True Positive Rate": y,
                            "Obesity Level": class_name,
                            "AUC": roc_auc_i
                        })
    
                roc_df = pd.DataFrame(roc_rows)
    
                fig = px.line(
                    roc_df,
                    x="False Positive Rate",
                    y="True Positive Rate",
                    color="Obesity Level",
                    title=f"Per-class ROC Curves — {eval_model_name}",
                    hover_data=["AUC"]
                )
    
                # Random classifier reference line
                fig.add_scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Random Classifier",
                    line=dict(dash="dot")
                )
    
                fig.update_xaxes(
                    range=[0, 1],
                    title="False Positive Rate"
                )
    
                fig.update_yaxes(
                    range=[0, 1],
                    title="True Positive Rate"
                )
    
                fig.update_layout(
                    height=550,
                    hovermode="closest"
                )
    
                st.plotly_chart(
                    fig,
                    use_container_width=True
                )
    
            else:
                st.info(
                    "This model does not expose class probabilities for ROC curves."
                )
    
        with st.expander("Full classification report"):
    
            report = classification_report(
                y_test,
                y_pred,
                target_names=label_encoder.classes_,
                zero_division=0,
                output_dict=True,
            )
    
            st.dataframe(
                pd.DataFrame(report).T.round(3),
                width="stretch"
            )

    # RANDOM FOREST — FEATURE IMPORTANCE
    if eval_model_name == "Random Forest":

        st.subheader("Random Forest — Feature Importance")

        feature_names = (
            eval_pipeline
            .named_steps["preprocessor"]
            .get_feature_names_out()
        )

        importances = (
            eval_pipeline
            .named_steps["classifier"]
            .feature_importances_
        )

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        })

        importance_df = (
            importance_df
            .sort_values("Importance", ascending=False)
            .head(15)
            .sort_values("Importance", ascending=True)
        )

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 15 Random Forest Feature Importances",
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


st.divider()

st.caption(
    "BMDS2003 Data Science — Group Project | "
    "CRISP-DM Prototype | Built with Streamlit"
)
