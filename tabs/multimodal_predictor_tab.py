
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def run():
    st.title("🦷 AI-Powered MBL Predictor with Simulation & Recommender")

    # Sample training data for demo purposes
    X_train = pd.DataFrame({
        'crown_type': ['PEKK', 'LD', 'PEKK', 'LD'],
        'gender': ['Female', 'Male', 'Female', 'Female'],
        'implant_site': ['Central', 'Lateral', 'Central', 'Central'],
        'baseline_HU': [1179, 695, 826, 954],
        'delta_HU': [-335, -126, 142, -232]
    })
    y_train = [0.6, 0.0, 1.9, 0.75]

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first'), ['crown_type', 'gender', 'implant_site']),
        ('num', StandardScaler(), ['baseline_HU', 'delta_HU'])
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    model.fit(X_train, y_train)

    st.sidebar.header("Enter Patient Features:")
    crown_type = st.sidebar.selectbox("Crown Type", ['PEKK', 'LD'])
    gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
    implant_site = st.sidebar.selectbox("Implant Site", ['Central', 'Lateral'])
    baseline_HU = st.sidebar.slider("Baseline HU", 400, 1600, 800)
    delta_HU = st.sidebar.slider("Change in HU", -500, 500, 0)

    user_input = pd.DataFrame([{
        'crown_type': crown_type,
        'gender': gender,
        'implant_site': implant_site,
        'baseline_HU': baseline_HU,
        'delta_HU': delta_HU
    }])

    prediction = model.predict(user_input)[0]
    st.subheader(f"📈 Predicted MBL: {prediction:.2f} mm")

    # MBL progression simulation
    months = np.array([3, 6, 12])
    progression = prediction + 0.05 * months
    st.line_chart(pd.DataFrame({'MBL (mm)': progression}, index=[f"{m} mo" for m in months]))

    # Treatment recommendation logic
    st.subheader("🦷 Treatment Recommendation")
    recommendation = "PEKK crown with delayed loading" if prediction > 1.0 else "LD crown with early loading"
    st.success(f"Recommended: {recommendation}")

    # SHAP explainability
    explainer = shap.Explainer(model.named_steps["regressor"])
    transformed_input = model.named_steps["preprocessor"].transform(user_input)
    shap_values = explainer(transformed_input)
    st.subheader("🔍 SHAP Feature Contribution")
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(bbox_inches="tight", dpi=300)

    # Visual Risk Radar
    st.subheader("🧭 Risk Factor Radar")
    risk_factors = {
        "Baseline HU": min(baseline_HU / 1600, 1),
        "Δ HU": min(abs(delta_HU) / 500, 1),
        "Crown Risk": 1 if crown_type == "PEKK" else 0.5,
        "Gender Risk": 0.3 if gender == "Female" else 0.7,
        "Site Risk": 0.6 if implant_site == "Lateral" else 0.4
    }
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    labels = list(risk_factors.keys())
    values = list(risk_factors.values())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    ax.fill(angles, values, color='skyblue', alpha=0.6)
    ax.plot(angles, values, color='blue')
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    st.pyplot(fig)
