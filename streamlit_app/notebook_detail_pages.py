"""Extra Streamlit pages derived from the latest report notebook.

The main explorer already provides selectable histograms, scatterplots,
boxplots, a numeric correlation heatmap, and a two-dimensional OLAP pivot.
This module therefore renders only the notebook EDA that those controls do
not reproduce.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def _insight(text):
    st.markdown(
        f'<div class="insight-card"><b>Interpretation:</b> {text}</div>',
        unsafe_allow_html=True,
    )


def _section(title):
    """Add visual breathing room and a clear boundary before each EDA section."""
    st.markdown('<div class="eda-section-gap"></div>', unsafe_allow_html=True)
    st.subheader(title)


def _cramers_v(feature, target):
    observed = pd.crosstab(feature, target).to_numpy(dtype=float)
    n = observed.sum()
    if n <= 1 or min(observed.shape) <= 1:
        return np.nan
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    chi_squared = np.sum((observed - expected) ** 2 / np.where(expected == 0, 1, expected))
    phi_squared = max(
        0,
        chi_squared / n
        - ((observed.shape[0] - 1) * (observed.shape[1] - 1)) / (n - 1),
    )
    corrected_rows = observed.shape[0] - ((observed.shape[0] - 1) ** 2) / (n - 1)
    corrected_columns = observed.shape[1] - ((observed.shape[1] - 1) ** 2) / (n - 1)
    denominator = min(corrected_rows - 1, corrected_columns - 1)
    return np.sqrt(phi_squared / denominator) if denominator > 0 else np.nan


def render_notebook_eda_page(
    df,
    obesity_order,
    obesity_colors,
    metadata,
    go_to_page,
    explore_page_name,
):
    """Render the unique EDA figures from the latest report notebook."""

    st.header("🧭 More EDA")
    st.caption(
        "Explore additional interactive patterns from the project analysis. "
        "Hover, zoom and select legend items to inspect each chart."
    )
    st.button(
        "← Return to interactive Data Exploration",
        key="back_to_explore_link",
        on_click=go_to_page,
        args=(explore_page_name,),
    )

    short_labels = {
        "Insufficient_Weight": "Insufficient",
        "Normal_Weight": "Normal",
        "Overweight_Level_I": "Overweight I",
        "Overweight_Level_II": "Overweight II",
        "Obesity_Type_I": "Obesity I",
        "Obesity_Type_II": "Obesity II",
        "Obesity_Type_III": "Obesity III",
    }
    severity_mapping = {label: index for index, label in enumerate(obesity_order)}
    eda_df = df.copy()
    eda_df["Severity_Rank"] = eda_df["Obesity_Level"].map(severity_mapping)

    # 5.3 Gender composition across obesity classes.
    _section("Gender composition across obesity classes")
    gender_counts = (
        pd.crosstab(eda_df["Obesity_Level"], eda_df["Gender"])
        .reindex(obesity_order, fill_value=0)
    )
    gender_percentages = gender_counts.div(gender_counts.sum(axis=1), axis=0).mul(100)
    gender_count_long = (
        gender_counts.rename(index=short_labels)
        .reset_index()
        .melt(id_vars="Obesity_Level", var_name="Gender", value_name="Records")
    )
    gender_share_long = (
        gender_percentages.rename(index=short_labels)
        .reset_index()
        .melt(id_vars="Obesity_Level", var_name="Gender", value_name="Share")
    )
    gender_col_1, gender_col_2 = st.columns(2)
    with gender_col_1:
        figure = px.bar(
            gender_count_long,
            x="Obesity_Level",
            y="Records",
            color="Gender",
            barmode="group",
            title="Gender counts within each class",
            color_discrete_sequence=["#DD8452", "#4C72B0"],
        )
        figure.update_xaxes(tickangle=-30, title="Obesity class")
        figure.update_layout(height=500, legend=dict(orientation="h", y=-0.35, x=.5, xanchor="center"), margin=dict(b=130))
        st.plotly_chart(figure, use_container_width=True)
    with gender_col_2:
        figure = px.bar(
            gender_share_long,
            x="Obesity_Level",
            y="Share",
            color="Gender",
            barmode="stack",
            title="Gender share within each class",
            color_discrete_sequence=["#DD8452", "#4C72B0"],
        )
        figure.update_yaxes(range=[0, 100], ticksuffix="%")
        figure.update_xaxes(tickangle=-30, title="Obesity class")
        figure.update_layout(height=500, legend=dict(orientation="h", y=-0.35, x=.5, xanchor="center"), margin=dict(b=130))
        st.plotly_chart(figure, use_container_width=True)
    _insight(
        "The count chart preserves sample size, while the proportional chart reveals "
        "class-specific gender imbalance. Extreme proportions in a class may influence "
        "model behaviour even when the full dataset is nearly gender-balanced."
    )

    # 5.7 Pearson versus Spearman.
    _section("Pearson versus Spearman association")
    severity_columns = [
        "Age", "Height", "Weight", "BMI", "Vegetable_Consumption_Freq",
        "Main_Meals_Per_Day", "Daily_Water_Intake", "Physical_Activity_Freq",
        "Technology_Usage_Time",
    ]
    association_rows = []
    for feature in severity_columns:
        association_rows.extend([
            {
                "Feature": feature.replace("_", " "),
                "Method": "Pearson",
                "Correlation": eda_df[feature].corr(eda_df["Severity_Rank"]),
            },
            {
                "Feature": feature.replace("_", " "),
                "Method": "Spearman",
                "Correlation": eda_df[feature].rank().corr(eda_df["Severity_Rank"].rank()),
            },
        ])
    severity_association = pd.DataFrame(association_rows)
    feature_order = (
        severity_association.query("Method == 'Spearman'")
        .sort_values("Correlation")["Feature"].tolist()
    )
    figure = px.bar(
        severity_association,
        x="Correlation",
        y="Feature",
        color="Method",
        barmode="group",
        category_orders={"Feature": feature_order},
        color_discrete_map={"Pearson": "#4C72B0", "Spearman": "#DD8452"},
        title="Numeric association with ordered obesity severity",
    )
    figure.add_vline(x=0, line_width=1, line_color="#475569")
    figure.update_layout(height=560, legend=dict(orientation="h", y=-.12, x=.5, xanchor="center"))
    st.plotly_chart(figure, use_container_width=True)
    _insight(
        "Pearson measures a straight-line relationship; Spearman measures a consistent "
        "increasing or decreasing rank pattern. Similar bars suggest a roughly linear trend, "
        "while a gap may indicate curvature or outlier sensitivity. Neither proves causation."
    )

    # 5.8 Lifestyle profiles.
    _section("Lifestyle profile by obesity class")
    lifestyle_columns = [
        "Vegetable_Consumption_Freq", "Main_Meals_Per_Day", "Daily_Water_Intake",
        "Physical_Activity_Freq", "Technology_Usage_Time",
    ]
    lifestyle_labels = {
        "Vegetable_Consumption_Freq": "Vegetables",
        "Main_Meals_Per_Day": "Main meals",
        "Daily_Water_Intake": "Water",
        "Physical_Activity_Freq": "Activity",
        "Technology_Usage_Time": "Technology",
    }
    lifestyle_ranges = {
        "Vegetable_Consumption_Freq": (1, 3), "Main_Meals_Per_Day": (1, 4),
        "Daily_Water_Intake": (1, 3), "Physical_Activity_Freq": (0, 3),
        "Technology_Usage_Time": (0, 2),
    }
    medians = eda_df.groupby("Obesity_Level", observed=True)[lifestyle_columns].median().reindex(obesity_order)
    display_values = medians.copy()
    for feature, (low, high) in lifestyle_ranges.items():
        display_values[feature] = (display_values[feature] - low) / (high - low)
    display_values = display_values.rename(index=short_labels, columns=lifestyle_labels)
    raw_medians = medians.rename(index=short_labels, columns=lifestyle_labels)
    figure = px.imshow(
        display_values,
        text_auto=False,
        zmin=0,
        zmax=1,
        color_continuous_scale="YlGnBu",
        labels={"color": "Position within scale"},
        title="Lifestyle profiles (colour normalised; hover shows raw median)",
    )
    figure.update_traces(
        customdata=raw_medians.to_numpy(),
        hovertemplate="<b>%{y}</b><br>%{x}<br>Raw median: %{customdata:.2f}<br>Scale position: %{z:.2f}<extra></extra>",
    )
    figure.update_yaxes(title="Obesity class")
    figure.update_layout(height=540)
    st.plotly_chart(figure, use_container_width=True)
    _insight(
        "Colour is normalised within each survey scale so variables measured from 0–2, "
        "0–3 and 1–4 can be compared visually. Hover values remain the raw class medians; "
        "the chart describes typical profiles rather than individual behaviour."
    )

    # 5.9 Cramer's V.
    _section("Bias-corrected Cramer’s V ranking")
    categorical_features = [
        "Gender", "Family_History_Overweight", "Frequent_High_Caloric_Food",
        "Food_Between_Meals", "Smokes", "Calorie_Monitoring",
        "Alcohol_Consumption", "Transportation_Mode", "Age_Group",
    ]
    cramer_rows = []
    for feature in categorical_features:
        if feature not in eda_df.columns:
            continue
        table = pd.crosstab(eda_df[feature], eda_df["Obesity_Level"])
        cramer_rows.append({
            "Feature": feature.replace("_", " "),
            "Cramer's V": _cramers_v(eda_df[feature], eda_df["Obesity_Level"]),
            "Smallest group": int(table.sum(axis=1).min()),
            "Categories": table.shape[0],
        })
    cramer_df = pd.DataFrame(cramer_rows).sort_values("Cramer's V")
    figure = px.bar(
        cramer_df,
        x="Cramer's V",
        y="Feature",
        orientation="h",
        color="Cramer's V",
        color_continuous_scale="Tealgrn",
        text="Cramer's V",
        hover_data={"Smallest group": True, "Categories": True, "Cramer's V": ":.4f"},
        title="Categorical association with obesity class",
    )
    figure.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    figure.update_xaxes(range=[0, 1])
    figure.update_layout(height=540, coloraxis_showscale=False)
    st.plotly_chart(figure, use_container_width=True)
    _insight(
        "Cramer’s V measures association magnitude from 0 toward 1 without assigning "
        "numeric distance to category names. It is not feature importance, does not show "
        "direction, and small groups can still make a pattern unstable."
    )

    # 5.10 Within-category composition.
    _section("Within-category obesity-class composition")
    composition_features = [
        "Family_History_Overweight", "Frequent_High_Caloric_Food",
        "Food_Between_Meals", "Transportation_Mode",
    ]
    for start in range(0, len(composition_features), 2):
        columns = st.columns(2)
        for container, feature in zip(columns, composition_features[start:start + 2]):
            with container:
                percentages = (
                    pd.crosstab(eda_df[feature], eda_df["Obesity_Level"], normalize="index")
                    .mul(100).reindex(columns=obesity_order, fill_value=0)
                    .rename(columns=short_labels).reset_index()
                    .melt(id_vars=feature, var_name="Obesity class", value_name="Share")
                )
                figure = px.bar(
                    percentages, x=feature, y="Share", color="Obesity class",
                    barmode="stack", title=feature.replace("_", " "),
                    color_discrete_sequence=[obesity_colors[level] for level in obesity_order],
                )
                figure.update_yaxes(range=[0, 100], ticksuffix="%")
                figure.update_xaxes(tickangle=-20)
                figure.update_layout(
                    height=480,
                    legend=dict(orientation="h", y=-.34, x=.5, xanchor="center", font=dict(size=9)),
                    margin=dict(b=135),
                )
                st.plotly_chart(figure, use_container_width=True)
    _insight(
        "Every bar totals 100% within its own category. The view reveals changes in class "
        "mix without allowing a large response group to appear important merely because it "
        "contains more records."
    )

    # 5.11 Water and activity.
    _section("Water intake by activity level")
    activity_band = pd.cut(
        eda_df["Physical_Activity_Freq"],
        bins=[-0.01, 1, 2, 3.01],
        labels=["Low activity", "Medium activity", "High activity"],
    )
    water_activity = (
        eda_df.assign(Activity_Band=activity_band)
        .groupby("Activity_Band", observed=True)["Daily_Water_Intake"]
        .agg(Records="size", Mean="mean", SD="std").reset_index()
    )
    water_activity["95% CI"] = 1.96 * water_activity["SD"] / np.sqrt(water_activity["Records"])
    figure = px.bar(
        water_activity, x="Activity_Band", y="Mean", error_y="95% CI",
        color="Activity_Band", text="Mean", hover_data={"Records": True, "95% CI": ":.3f"},
        title="Mean daily water intake by activity band",
        color_discrete_sequence=["#457B9D", "#2A9D8F", "#E9C46A"],
    )
    figure.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    figure.update_yaxes(title="Daily water intake score")
    figure.update_layout(height=470, showlegend=False)
    st.plotly_chart(figure, use_container_width=True)
    _insight(
        "Bars compare group means and capped lines show approximate 95% confidence intervals. "
        "Hover also reveals group size. This avoids the misleading water-intake boxplot and "
        "does not imply that activity causes water intake to change."
    )

    # 5.12 Interaction.
    _section("Activity and technology-use interaction")
    technology_band = pd.cut(
        eda_df["Technology_Usage_Time"],
        bins=[-0.01, 0.75, 1.5, 2.01],
        labels=["Low technology", "Medium technology", "High technology"],
    )
    interaction = pd.DataFrame({
        "Activity band": activity_band,
        "Technology band": technology_band,
        "Severity": eda_df["Severity_Rank"],
    }).dropna()
    interaction_summary = (
        interaction.groupby(["Activity band", "Technology band"], observed=False)["Severity"]
        .agg(Mean="mean", Records="size").reset_index()
    )
    figure = px.bar(
        interaction_summary,
        x="Activity band", y="Mean", color="Technology band", barmode="group",
        hover_data={"Records": True, "Mean": ":.3f"},
        title="Mean severity rank by activity and technology-use band",
        color_discrete_sequence=["#4C72B0", "#2A9D8F", "#E76F51"],
    )
    figure.update_yaxes(title="Mean severity rank (0 = lowest, 6 = highest)")
    figure.update_layout(height=490, legend=dict(orientation="h", y=-.20, x=.5, xanchor="center"), margin=dict(b=95))
    st.plotly_chart(figure, use_container_width=True)
    _insight(
        "Grouped bars reveal whether activity and technology-use bands jointly correspond "
        "to different average severity. Hover counts expose sparse combinations before a "
        "large-looking difference is treated as meaningful."
    )

    # 5.13 Age composition.
    _section("Age-group and class composition")
    age_counts = pd.crosstab(eda_df["Age_Group"], eda_df["Obesity_Level"]).reindex(columns=obesity_order, fill_value=0)
    age_percentages = age_counts.div(age_counts.sum(axis=1), axis=0).mul(100).rename(columns=short_labels)
    figure = px.imshow(
        age_percentages,
        text_auto=".1f",
        zmin=0, zmax=100,
        color_continuous_scale="YlOrRd",
        labels={"color": "Within-age-group share (%)"},
        title="Obesity-class composition within equal-width age groups",
    )
    figure.update_traces(hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>")
    figure.update_xaxes(tickangle=-25, title="Obesity class")
    figure.update_yaxes(title="Age group")
    figure.update_layout(height=520, margin=dict(b=100))
    st.plotly_chart(figure, use_container_width=True)
    _insight(
        "Rows sum to 100%, so the chart compares class composition inside each engineered "
        "age band. Age groups are used only for EDA and are not additional model inputs."
    )

    # 5.14 Body measurements by gender.
    _section("Body measurements by gender")
    body_col_1, body_col_2 = st.columns(2)
    for container, feature, axis_label in zip(
        [body_col_1, body_col_2],
        ["Height", "Weight"],
        ["Height (m)", "Weight (kg)"],
    ):
        with container:
            figure = px.box(
                eda_df, x="Gender", y=feature, color="Gender", points=False,
                color_discrete_sequence=["#4C72B0", "#DD8452"],
                title=f"{axis_label} by gender",
            )
            figure.update_layout(height=470, showlegend=False)
            st.plotly_chart(figure, use_container_width=True)
    _insight(
        "The paired boxplots compare median, spread and central body-measurement ranges by "
        "gender. They describe this sample and should not be treated as universal population norms."
    )

    # 5.15 BMI ECDF.
    _section("BMI empirical cumulative distributions")
    figure = px.ecdf(
        eda_df,
        x="BMI",
        color="Obesity_Level",
        category_orders={"Obesity_Level": obesity_order},
        color_discrete_map=obesity_colors,
        title="Empirical cumulative distribution of BMI by class",
        labels={"proportion": "Proportion at or below BMI"},
    )
    figure.update_layout(height=520, legend=dict(orientation="h", y=-.25, x=.5, xanchor="center"), margin=dict(b=115))
    st.plotly_chart(figure, use_container_width=True)
    _insight(
        "At any BMI threshold, the vertical position gives the proportion of that class at "
        "or below the threshold. ECDFs avoid bin choices and make overlap between neighbouring "
        "classes visible."
    )

    # 5.16 Three-dimensional OLAP drill-down.
    _section("Gender × transport × obesity-class drill-down")
    drilldown = (
        pd.crosstab(
            [eda_df["Gender"], eda_df["Transportation_Mode"]],
            eda_df["Obesity_Level"],
        )
        .reindex(columns=obesity_order, fill_value=0)
        .reset_index()
        .melt(
            id_vars=["Gender", "Transportation_Mode"],
            var_name="Obesity_Level",
            value_name="Records",
        )
    )
    figure = px.bar(
        drilldown,
        x="Transportation_Mode", y="Records", color="Obesity_Level",
        facet_row="Gender", category_orders={"Obesity_Level": obesity_order},
        color_discrete_map=obesity_colors,
        title="Interactive OLAP drill-down: transport and gender",
    )
    figure.update_layout(height=650, legend=dict(orientation="h", y=-.20, x=.5, xanchor="center"), margin=dict(b=100))
    figure.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.replace("Gender=", "")))
    st.plotly_chart(figure, use_container_width=True)
    _insight(
        "The facets drill from gender into transportation mode and obesity class. Hover for "
        "counts and click legend entries to isolate a class. Very small transport groups should "
        "be interpreted cautiously."
    )
