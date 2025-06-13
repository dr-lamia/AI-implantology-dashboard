
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

    with st.form("form"):
        age = st.slider("Age", 18, 90, 45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        smoking = st.selectbox("Smoking", ["Yes", "No"])
        baseline_HU = st.slider("Baseline HU", 400, 1600, 950)
        implant_site = st.selectbox("Implant Site", ["Central", "Lateral"])
        crown_type = st.selectbox("Crown Type", ["PEKK", "LD"])
        loading_time = st.selectbox("Loading Time", ["Immediate", "Delayed"])
        submitted = st.form_submit_button("Generate Report")

    if submitted:
        input_df = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "smoking": smoking,
            "baseline_HU": baseline_HU,
            "implant_site": implant_site,
            "crown_type": crown_type,
            "loading_time": loading_time,
            "delta_HU": 0
        }])

        train_df = pd.DataFrame({
            "age": [45, 55, 35, 60],
            "gender": ["Male", "Female", "Female", "Male"],
            "smoking": ["Yes", "No", "Yes", "No"],
            "baseline_HU": [1000, 800, 1200, 950],
            "implant_site": ["Central", "Lateral", "Central", "Central"],
            "crown_type": ["PEKK", "LD", "LD", "PEKK"],
            "loading_time": ["Immediate", "Delayed", "Immediate", "Delayed"],
            "delta_HU": [50, -100, 75, -80]
        })
        y = [1.2, 0.4, 0.8, 1.0]

        pre = ColumnTransformer([
            ("cat", OneHotEncoder(drop="first"), ["gender", "smoking", "implant_site", "crown_type", "loading_time"]),
            ("num", StandardScaler(), ["age", "baseline_HU", "delta_HU"])
        ])

        model = Pipeline([
            ("pre", pre),
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        model.fit(train_df, y)
        pred = model.predict(input_df)[0]
        st.metric("Predicted MBL", f"{pred:.2f} mm")

        progression = pred + np.array([0.05 * i for i in [3, 6, 12]])
        st.line_chart(pd.DataFrame({"MBL": progression}, index=["3 mo", "6 mo", "12 mo"]))

        recommendation = "PEKK crown with delayed loading" if pred > 1.0 or smoking == "Yes" else "LD crown with immediate loading"
        st.success(f"Treatment Recommendation: {recommendation}")

        # SHAP safe plotting
        shap_path = None
        try:
            explainer = shap.Explainer(model.named_steps["rf"])
            X_trans = model.named_steps["pre"].transform(input_df)
            shap_vals = explainer(X_trans)
            shap_array = shap_vals.values[0]
            feature_names = shap_vals.feature_names
            if hasattr(shap_array, '__len__') and len(feature_names) == len(shap_array):
                sorted_pairs = sorted(zip(np.abs(shap_array), feature_names, shap_array), reverse=True)
                labels = [label for _, label, _ in sorted_pairs]
                values = [val for _, _, val in sorted_pairs]
                fig, ax = plt.subplots()
                ax.barh(labels[::-1], values[::-1], color='salmon')
                ax.set_title("SHAP Top Contributions")
                plt.tight_layout()
                shap_path = "/tmp/shap_safe.png"
                plt.savefig(shap_path)
                plt.close()
            else:
                st.warning("Mismatch in SHAP outputs.")
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
        radar_path = "/tmp/radar_safe.png"
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
                self.cell(0, 10, f"Predicted MBL: {pred:.2f} mm", ln=True)
                self.cell(0, 10, f"Recommendation: {recommendation}", ln=True)
                for m, v in zip(["3 mo", "6 mo", "12 mo"], progression):
                    self.cell(0, 10, f"{m}: {v:.2f} mm", ln=True)
                self.ln(5)
                if shap_path:
                    self.cell(0, 10, "Top SHAP Features:", ln=True)
                    self.image(shap_path, w=120)
                    self.multi_cell(0, 10, "These features most influenced the prediction. Longer bars indicate greater contribution.")
                self.ln(3)
                self.cell(0, 10, "Risk Factor Radar:", ln=True)
                self.image(radar_path, w=120)
                self.multi_cell(0, 10, "Radar chart shows modifiable and non-modifiable risks. Wider shape = greater total risk.")

        pdf = PDF()
        pdf.add_page()
        pdf.content()
        pdf_path = "/mnt/data/Dr_ElFadaly_Explainability_Report_Safe.pdf"
        pdf.output(pdf_path)
        st.download_button("📥 Download PDF Report", data=open(pdf_path, "rb"), file_name="MBL_Report_ElFadaly.pdf")
