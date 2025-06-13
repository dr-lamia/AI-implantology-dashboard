
import streamlit as st

def run():
    st.title("💬 Feedback & Clinical Insights")
    st.write("Submit your feedback to help improve this AI tool.")
    st.text_area("Your thoughts", placeholder="What worked well? What can be improved?")
    st.slider("How useful is this dashboard?", 1, 5)
    st.button("Submit Feedback")
