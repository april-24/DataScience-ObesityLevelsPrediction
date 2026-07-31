import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="ObePredict System", layout="wide")

# Custom CSS to make it look professional
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #2C3E50;}
    .sub-header {font-size: 1.5rem; font-weight: bold; color: #34495E;}
    </style>
""", unsafe_allow_html=True)

# Define the Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About", "Data Exploration", "Model Performance", "Prediction"])

# Load Dataset (cached for performance)
@st.cache_data
def load_data():
    # Replace with your actual dataset path
    df = pd.read_csv("obesity.csv")
    return df

df = load_data()

# ==========================================
# PAGE 1: ABOUT
# ==========================================
if page == "About":
    st.markdown('<div class="main-header">About the Project</div>', unsafe_allow_html=True)
    st.write("""
    Welcome to **ObePredict**, an intelligent system designed to estimate obesity levels based on eating habits and physical condition.
    
    ### Business Problem
    Obesity is a growing global health concern. This project aims to utilize machine learning algorithms to predict an individual's obesity risk level, allowing for early intervention and personalized health recommendations.
    
    ### Team Members
    *   **Kyra** - Baseline Model (Decision Tree)
    *   **Liping** - Random Forest
    *   **Wenhsuan** - Support Vector Machine (SVM)
    *   **Gladys** - K-Nearest Neighbors (KNN)
    
    ### Dataset
    The dataset contains 17 attributes and 2111 records, focusing on eating habits (e.g., frequency of vegetable consumption, high-caloric food intake) and physical conditions.
    """)

# ==========================================
# PAGE 2: DATA EXPLORATION
# ==========================================
elif page == "Data Exploration":
    st.markdown('<div class="main-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    st.write("### Raw Dataset Overview")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Obesity Levels Distribution")
        fig, ax = plt.subplots(figsize=(6,4))
        obesity_counts = df['NObeyesdad'].value_counts()
        ax.pie(obesity_counts, labels=obesity_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
        
    with col2:
        st.write("### Age Histogram")
        fig2, ax2 = plt.subplots(figsize=(6,4))
        sns.histplot(df['Age'], bins=20, kde=True, ax=ax2)
        st.pyplot(fig2)
        
    st.write("### Height vs Weight Scatterplot")
    fig3, ax3 = plt.subplots(figsize=(8,4))
    sns.scatterplot(data=df, x='Height', y='Weight', hue='NObeyesdad', ax=ax3)
    st.pyplot(fig3)

# ==========================================
# PAGE 3: MODEL PERFORMANCE
# ==========================================
elif page == "Model Performance":
    st.markdown('<div class="main-header">Model Evaluation & Comparison</div>', unsafe_allow_html=True)
    st.write("Compare the performance of the four implemented machine learning models.")
    
    # Example placeholder data for the models
    metrics_data = {
        "Model": ["Decision Tree (Kyra)", "Random Forest (Liping)", "SVM (Wenhsuan)", "KNN (Gladys)"],
        "Accuracy": ["92.5%", "96.8%", "94.2%", "89.5%"],
        "Precision": ["91.8%", "96.5%", "93.9%", "88.7%"],
        "Recall": ["92.1%", "96.7%", "94.1%", "89.0%"],
        "F1-Score": ["91.9%", "96.6%", "94.0%", "88.8%"]
    }
    metrics_df = pd.DataFrame(metrics_data)
    st.table(metrics_df)
    
    st.info("Note: Random Forest yielded the highest accuracy and is selected as the primary model for deployment.")

# ==========================================
# PAGE 4: PREDICTION
# ==========================================
elif page == "Prediction":
    st.markdown('<div class="main-header">Obesity Level Predictor</div>', unsafe_allow_html=True)
    st.write("Enter the patient's details below to estimate their obesity risk level.")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=10, max_value=100, value=25)
            height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.70)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
            family_history = st.selectbox("Family History of Overweight?", ["yes", "no"])
            
        with col2:
            favc = st.selectbox("Frequent High-Caloric Food? (FAVC)", ["yes", "no"])
            fcvc = st.slider("Vegetables in meals? (FCVC) [1-3]", 1, 3, 2)
            ncp = st.slider("Main meals a day? (NCP) [1-4]", 1, 4, 3)
            caec = st.selectbox("Eat between meals? (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
            smoke = st.selectbox("Do you smoke?", ["yes", "no"])
            
        with col3:
            ch2o = st.slider("Water intake (L)? (CH2O) [1-3]", 1, 3, 2)
            scc = st.selectbox("Monitor calories? (SCC)", ["yes", "no"])
            faf = st.slider("Physical activity frequency? (FAF) [0-3]", 0, 3, 1)
            tue = st.slider("Tech device time? (TUE) [0-2]", 0, 2, 1)
            calc = st.selectbox("Alcohol consumption? (CALC)", ["no", "Sometimes", "Frequently", "Always"])
            mtrans = st.selectbox("Transportation", ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])
            
        submit_button = st.form_submit_button("Predict Obesity Level")
        
    if submit_button:
        # In a real app, you would load your saved .pkl model here and pass this data.
        # e.g., model = joblib.load('random_forest_model.pkl')
        # prediction = model.predict(input_data)
        
        st.success("Prediction Complete!")
        # Mock result for visual prototype
        st.metric(label="Estimated Obesity Level", value="Normal_Weight")
        st.write("*(Disclaimer: This is a placeholder prediction for the prototype. Connect your trained Random Forest model to generate live predictions.)*")
