
import streamlit as st

def run():
    st.title("📂 Dataset Builder")
    st.write("Upload structured implantology records to contribute to the shared training dataset.")
    st.file_uploader("Upload anonymized .csv file", type="csv")
    st.button("Submit to Dataset")
