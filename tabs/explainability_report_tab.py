
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import os

def run():
    st.title("📄 Explainability Report Generator")

    with st.form("report_form"):
        age = st.number_input("Age", min_value=18, max_value=90, value=45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        smoking = st.selectbox("Smoking", ["Yes", "No"])
        baseline_HU = st.slider("Baseline HU", 400, 1600, 950)
        implant_site = st.selectbox("Implant Site", ["Central", "Lateral"])
        crown_type = st.selectbox("Crown Type", ["PEKK", "LD"])
        loading_time = st.selectbox("Loading Time", ["Immediate", "Delayed"])
        submitted = st.form_submit_button("Generate Report")

    if submitted:
        df_input = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "smoking": smoking,
            "baseline_HU": baseline_HU,
            "implant_site": implant_site,
            "crown_type": crown_type,
            "loading_time": loading_time,
            "delta_HU": 0
        }])

        df_train = pd.DataFrame({
            "age": [40, 60, 50, 30],
            "gender": ["Male", "Female", "Female", "Male"],
            "smoking": ["Yes", "No", "Yes", "No"],
            "baseline_HU": [950, 1100, 850, 980],
            "implant_site": ["Central", "Lateral", "Central", "Central"],
            "crown_type": ["PEKK", "LD", "LD", "PEKK"],
            "loading_time": ["Immediate", "Delayed", "Immediate", "Delayed"],
            "delta_HU": [-100, 50, -150, 75]
        })
        y_train = [1.2, 0.3, 0.7, 1.1]

        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(drop='first'), ["gender", "smoking", "implant_site", "crown_type", "loading_time"]),
            ('num', StandardScaler(), ["age", "baseline_HU", "delta_HU"])
        ])

        model = Pipeline([
            ("pre", preprocessor),
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        model.fit(df_train, y_train)

        pred = model.predict(df_input)[0]
        st.metric("Predicted MBL (mm)", f"{pred:.2f}")
        months = np.array([3, 6, 12])
        progression = pred + 0.05 * months

        st.line_chart(pd.DataFrame({"MBL (mm)": progression}, index=["3 mo", "6 mo", "12 mo"]))

        recommendation = "PEKK crown with delayed loading" if pred > 1.0 or smoking == "Yes" else "LD crown with immediate loading"
        st.success(f"Treatment Recommendation: {recommendation}")

        explainer = shap.Explainer(model.named_steps["rf"])
        transformed_input = model.named_steps["pre"].transform(df_input)
        shap_vals = explainer(transformed_input)

        fig_radar, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        risk_factors = {
            "Baseline HU": min(baseline_HU / 1600, 1),
            "Smoking Risk": 0.8 if smoking == "Yes" else 0.3,
            "Crown Risk": 1 if crown_type == "PEKK" else 0.5,
            "Loading Risk": 0.6 if loading_time == "Immediate" else 0.3
        }
        labels = list(risk_factors.keys())
        values = list(risk_factors.values())
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        ax.fill(angles, values, color='skyblue', alpha=0.6)
        ax.plot(angles, values, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        radar_img = "/tmp/radar_elfadaly.png"
        fig_radar.savefig(radar_img)
        plt.close()

        fig_shap, ax_shap = plt.subplots()
        shap_vals_array = shap_vals.values[0]
        feature_names = shap_vals.feature_names
        sorted_idx = np.argsort(np.abs(shap_vals_array))[::-1]
        ax_shap.barh(np.array(feature_names)[sorted_idx], shap_vals_array[sorted_idx], color='salmon')
        ax_shap.set_title("SHAP Top Contributions")
        shap_img = "/tmp/shap_elfadaly.png"
        plt.tight_layout()
        fig_shap.savefig(shap_img)
        plt.close()

        class PDF(FPDF):
            def header(self):
                logo_path = "A_logo_for_the_\"Dr._ElFadaly_AI_Implantology_Dashb.png"
                if os.path.exists(logo_path):
                    self.image(logo_path, x=10, y=8, w=35)
                self.set_font("Arial", "B", 14)
                self.cell(0, 10, "Dr. ElFadaly - AI Implantology Dashboard", ln=True, align="C")
                self.ln(15)
            def footer(self):
                self.set_y(-15)
                self.set_font("Arial", "I", 8)
                self.cell(0, 10, f"Page {self.page_no()}", align="C")
            def add_content(self):
                self.set_font("Arial", "", 12)
                self.cell(0, 10, f"Predicted MBL: {pred:.2f} mm", ln=True)
                self.cell(0, 10, f"Recommendation: {recommendation}", ln=True)
                for m, v in zip(["3 mo", "6 mo", "12 mo"], progression):
                    self.cell(0, 10, f"{m}: {v:.2f} mm", ln=True)
                self.ln(5)
                self.cell(0, 10, "Top SHAP Feature Contributions:", ln=True)
                self.image(shap_img, w=120)
                self.multi_cell(0, 10, "SHAP highlights how input features contribute to the model's prediction. Red bars indicate features increasing risk.")
                self.ln(3)
                self.cell(0, 10, "Risk Factor Radar:", ln=True)
                self.image(radar_img, w=120)
                self.multi_cell(0, 10, "Radar chart shows modifiable and non-modifiable risks. Higher coverage = greater cumulative risk.")

        pdf = PDF()
        pdf.add_page()
        pdf.add_content()
        report_path = "/mnt/data/Dr_ElFadaly_Explainability_Report.pdf"
        pdf.output(report_path)
        st.download_button("📥 Download PDF Report", data=open(report_path, "rb"), file_name="ElFadaly_MBL_Report.pdf")
