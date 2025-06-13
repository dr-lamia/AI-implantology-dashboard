
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
    st.title("📄 Explainability Report Generator (Synthetic-trained Model)")

    with st.form("form"):
        age = st.slider("Age", 18, 90, 45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        smoking = st.selectbox("Smoking", ["Yes", "No"])
        baseline_HU = st.slider("Baseline HU", 400, 1600, 950)
        implant_site = st.selectbox("Implant Site", ["Central", "Lateral"])
        crown_type = st.selectbox("Crown Type", ["PEKK", "LD"])
        loading_time = st.selectbox("Loading Time", ["Immediate", "Early", "Conventional"])
        submitted = st.form_submit_button("Generate Report")

    if submitted:
        input_df = pd.DataFrame([{
            "HU_baseline": baseline_HU,
            "Crown": crown_type,
            "Gender": gender,
            "Implant Site": implant_site,
            "Age": age,
            "Smoking": smoking,
            "Loading_Time": loading_time
        }])

        data = pd.read_csv("synthetic_mbl_dataset_with_clinical_factors.csv")
        features = ["HU_baseline", "Crown", "Gender", "Implant Site", "Age", "Smoking", "Loading_Time"]
        target = "MBL"

        X = data[features]
        y = data[target]

        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(drop="first"), ["Crown", "Gender", "Implant Site", "Smoking", "Loading_Time"]),
            ("num", StandardScaler(), ["HU_baseline", "Age"])
        ])

        model = Pipeline([
            ("pre", preprocessor),
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        model.fit(X, y)
        prediction = model.predict(input_df)[0]
        st.metric("Predicted MBL", f"{prediction:.2f} mm")

        progression = prediction + np.array([0.05 * i for i in [3, 6, 12]])
        st.line_chart(pd.DataFrame({"MBL": progression}, index=["3 mo", "6 mo", "12 mo"]))

        recommendation = "PEKK crown with delayed loading" if prediction > 1.0 or smoking == "Yes" else "LD crown with immediate loading"
        st.success(f"Treatment Recommendation: {recommendation}")

        shap_path, radar_path = None, None
        try:
            explainer = shap.Explainer(model.named_steps["rf"])
            X_trans = model.named_steps["pre"].transform(input_df)
            shap_vals = explainer(X_trans)
            shap_array = shap_vals.values[0]
            feature_names = shap_vals.feature_names or model.named_steps["pre"].get_feature_names_out()
            if hasattr(shap_array, '__len__') and len(feature_names) == len(shap_array):
                sorted_pairs = sorted(zip(np.abs(shap_array), feature_names, shap_array), reverse=True)
                labels = [label for _, label, _ in sorted_pairs]
                values = [val for _, _, val in sorted_pairs]
                fig, ax = plt.subplots()
                ax.barh(labels[::-1], values[::-1], color='salmon')
                ax.set_title("SHAP Top Contributions")
                plt.tight_layout()
                shap_path = "shap_output.png"
                plt.savefig(shap_path)
                plt.close()
        except Exception as e:
            st.error(f"SHAP Error: {e}")

        radar_vals = {
            "Baseline HU": min(baseline_HU / 1600, 1),
            "Smoking Risk": 0.8 if smoking == "Yes" else 0.3,
            "Crown Risk": 1 if crown_type == "PEKK" else 0.5,
            "Loading Risk": 0.6 if loading_time == "Immediate" else 0.3
        }
        radar_labels = list(radar_vals.keys())
        radar_values = list(radar_vals.values())
        radar_angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
        radar_values += radar_values[:1]
        radar_angles += radar_angles[:1]
        fig_radar, ax_radar = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        ax_radar.fill(radar_angles, radar_values, color='skyblue', alpha=0.6)
        ax_radar.plot(radar_angles, radar_values, color='blue')
        ax_radar.set_xticks(radar_angles[:-1])
        ax_radar.set_xticklabels(radar_labels)
        radar_path = "radar_output.png"
        plt.savefig(radar_path)
        plt.close()

        class PDF(FPDF):
            def header(self):
                logo_path = "A_logo_for_the_\"Dr._ElFadaly_AI_Implantology_Dashb.png"
                if os.path.exists(logo_path):
                    self.image(logo_path, x=10, y=8, w=35)
                self.set_font("Arial", "B", 14)
                self.cell(0, 10, "Dr. ElFadaly - AI Implantology Dashboard", ln=True, align="C")
                self.ln(10)
            def footer(self):
                self.set_y(-15)
                self.set_font("Arial", "I", 8)
                self.cell(0, 10, f"Page {self.page_no()}", align="C")
            def content(self):
                self.set_font("Arial", "", 12)
                self.cell(0, 10, f"Predicted MBL: {prediction:.2f} mm", ln=True)
                self.cell(0, 10, f"Recommendation: {recommendation}", ln=True)
                for m, v in zip(["3 mo", "6 mo", "12 mo"], progression):
                    self.cell(0, 10, f"{m}: {v:.2f} mm", ln=True)
                self.ln(5)
                if shap_path:
                    self.cell(0, 10, "Top SHAP Features:", ln=True)
                    self.image(shap_path, w=120)
                    self.multi_cell(0, 10, "These features most influenced the prediction. Longer bars = greater contribution.")
                self.ln(3)
                self.cell(0, 10, "Risk Factor Radar:", ln=True)
                self.image(radar_path, w=120)
                self.multi_cell(0, 10, "Radar chart shows modifiable and non-modifiable risks. Wider shape = greater total risk.")

        pdf = PDF()
        pdf.add_page()
        pdf.content()
        pdf_path = "Dr_ElFadaly_Explainability_Synthetic_Model.pdf"
        pdf.output(pdf_path)
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download PDF Report", data=f, file_name="MBL_Report_Synthetic_Model.pdf")
