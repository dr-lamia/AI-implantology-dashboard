import streamlit as st
import pandas as pd
import os
from datetime import datetime

st.title("💬 Feedback and Suggestions")

st.markdown("Help us improve the AI Implantology Dashboard. Share your experience or ideas!")

# Create feedback storage directory
FEEDBACK_FILE = "user_feedback.csv"
if not os.path.exists(FEEDBACK_FILE):
    df = pd.DataFrame(columns=["timestamp", "user_role", "rating", "comments"])
    df.to_csv(FEEDBACK_FILE, index=False)

# Input fields
user_role = st.selectbox("I am a...", ["Clinician", "Researcher", "Student", "Other"])
rating = st.slider("How satisfied are you with the app?", 1, 5, 3)
comments = st.text_area("Your feedback, comments, or suggestions:", height=150)

# Submit button
if st.button("Submit Feedback"):
    new_feedback = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_role": user_role,
        "rating": rating,
        "comments": comments
    }
    # Append to CSV
    df = pd.read_csv(FEEDBACK_FILE)
    df = pd.concat([df, pd.DataFrame([new_feedback])], ignore_index=True)
    df.to_csv(FEEDBACK_FILE, index=False)
    st.success("✅ Thank you for your feedback!")

# Display previous feedback if user is admin
with st.expander("📊 View Submitted Feedback"):
    df = pd.read_csv(FEEDBACK_FILE)
    st.dataframe(df.tail(10))
