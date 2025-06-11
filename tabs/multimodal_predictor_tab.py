import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Title
st.title("🧬 Multimodal MBL Predictor (CBCT + Clinical)")

# Image feature extraction
def extract_basic_image_features(image):
    image = image.resize((64, 64))
    img_array = np.array(image).mean(axis=2)
    return np.array([img_array.mean(), img_array.std()])

# Feature builder
def build_fusion_features(image, features):
    img_features = extract_basic_image_features(image)
    tab_features = np.array([[
        features['age'],
        1 if features['gender'] == 'Male' else 0,
        1 if features['smoking'] == 'Yes' else 0,
        features['baseline_HU'],
        1 if features['implant_site'] == 'Central' else 0,
        1 if features['crown_type'] == 'PEKK' else 0,
        1 if features['loading_time'] == 'Immediate' else 0
    ]])
    combined = np.concatenate([tab_features[0], img_features])
    return combined.reshape(1, -1)

# Train dummy model
def train_dummy_model():
    np.random.seed(42)
    X = np.random.rand(100, 9)
    y = 0.5 * X[:, 0] + 0.2 * X[:, 3] + 0.1 * X[:, 8] + np.random.normal(0, 0.2, 100)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)
    return model, scaler

model, scaler = train_dummy_model()

# Image upload
st.subheader("🖼️ Upload CBCT Image")
cbct_file = st.file_uploader("Upload CBCT Slice (JPG/PNG)", type=["jpg", "jpeg", "png"])

# Clinical input
st.subheader("📝 Enter Clinical Features")
age = st.number_input("Age", 18, 90, 45)
gender = st.selectbox("Gender", ["Male", "Female"])
smoking = st.selectbox("Smoking Status", ["No", "Yes"])
baseline_HU = st.slider("Baseline HU", 400, 1600, 800)
implant_site = st.selectbox("Implant Site", ["Central", "Lateral"])
crown_type = st.selectbox("Crown Type", ["PEKK", "LD"])
loading_time = st.selectbox("Loading Time", ["Immediate", "Delayed"])

# Prediction
if st.button("🔍 Predict MBL"):
    if cbct_file is not None:
        image = Image.open(cbct_file).convert("RGB")
        st.image(image, caption="CBCT Input", use_column_width=True)
        features = {
            'age': age,
            'gender': gender,
            'smoking': smoking,
            'baseline_HU': baseline_HU,
            'implant_site': implant_site,
            'crown_type': crown_type,
            'loading_time': loading_time
        }
        fused = build_fusion_features(image, features)
        scaled = scaler.transform(fused)
        pred = model.predict(scaled)[0]
        st.success(f"📈 Predicted MBL: {pred:.2f} mm")
    else:
        st.warning("⚠️ Please upload a CBCT image first.")
