import streamlit as st
from fpdf import FPDF
import datetime
import tempfile

class CleanMBLReportPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "AI-Powered MBL Prediction Report", ln=True, align="C")
        self.set_font("Arial", "", 10)
        self.cell(0, 10, f"Date: {datetime.date.today().strftime('%B %d, %Y')}", ln=True, align="C")
        self.ln(10)

    def section_title(self, title):
        self.set_font("Arial", "B", 12)
        self.set_text_color(30, 30, 120)
        self.cell(0, 10, title, ln=True)
        self.set_text_color(0, 0, 0)

    def section_body(self, text):
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 8, text)
        self.ln()

    def add_prediction_summary(self, features, mbl_value, top_features):
        self.section_title("Clinical Input Summary")
        summary = "\n".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in features.items()])
        self.section_body(summary)

        self.section_title("Predicted Marginal Bone Loss")
        self.section_body(f"Estimated MBL: {mbl_value:.2f} mm")

        if top_features:
            self.section_title("Top Contributing Factors")
            formatted = "\n".join([f"{feat}: {val:.3f}" for feat, val in top_features.items()])
            self.section_body(formatted)

        self.section_title("AI Clinical Interpretation")
        if mbl_value >= 1.0:
            self.section_body("High-risk case. Consider additional intervention or altered loading protocol.")
        elif mbl_value >= 0.5:
            self.section_body("Moderate bone loss predicted. Standard monitoring recommended.")
        else:
            self.section_body("Low risk of MBL. Proceed with standard treatment plan.")

st.title("📄 Explainability Report Generator")

with st.form("pdf_form"):
    age = st.number_input("Age", 18, 90, 45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking = st.selectbox("Smoking", ["Yes", "No"])
    baseline_HU = st.slider("Baseline HU", 400, 1600, 950)
    implant_site = st.selectbox("Implant Site", ["Central", "Lateral"])
    crown_type = st.selectbox("Crown Type", ["PEKK", "LD"])
    loading_time = st.selectbox("Loading Time", ["Immediate", "Delayed"])
    predicted_mbl = st.slider("Predicted MBL (mm)", 0.0, 3.0, 1.0, 0.01)

    top_contributors = st.text_area("Top Contributing Features (e.g. age: 0.12\nbaseline_HU: 0.25)", height=100)

    submit = st.form_submit_button("Generate PDF Report")

if submit:
    features = {
        "age": age,
        "gender": gender,
        "smoking": smoking,
        "baseline_HU": baseline_HU,
        "implant_site": implant_site,
        "crown_type": crown_type,
        "loading_time": loading_time
    }

    # Parse top contributors
    top_features = {}
    for line in top_contributors.strip().splitlines():
        try:
            k, v = line.split(":")
            top_features[k.strip()] = float(v.strip())
        except:
            continue

    pdf = CleanMBLReportPDF()
    pdf.add_page()
    pdf.add_prediction_summary(features, predicted_mbl, top_features)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        pdf.output(tmpfile.name)
        st.success("✅ PDF Report Generated!")
        with open(tmpfile.name, "rb") as f:
            st.download_button("📥 Download Report", f, file_name="MBL_Explainability_Report.pdf")
