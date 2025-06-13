
import streamlit as st
from tabs import (
    dataset_builder_tab,
    explainability_report_tab,
    feedback_tab,
    multimodal_predictor_tab
)

st.set_page_config(page_title="AI Implantology Dashboard", layout="wide")

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/8/88/Tooth_icon.png", width=50)
st.sidebar.title("AI Implantology Dashboard")

tabs = {
    "MBL Predictor & SHAP": multimodal_predictor_tab,
    "Explainability Report": explainability_report_tab,
    "Dataset Builder": dataset_builder_tab,
    "Feedback": feedback_tab,
}

selected_tab = st.sidebar.radio("Navigate", list(tabs.keys()))
tabs[selected_tab].run()
