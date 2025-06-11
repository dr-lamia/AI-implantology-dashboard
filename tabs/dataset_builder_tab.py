import streamlit as st
import pandas as pd
import os
from PIL import Image
import uuid

st.title("📂 Dataset Builder for MBL AI Training")

st.markdown("Upload anonymized patient data to grow the shared research dataset.")

# Create a dataset directory if it doesn't exist
DATASET_DIR = "shared_dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

# Upload CSV file
st.subheader("📝 Upload Clinical Data (CSV)")
csv_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Upload images
st.subheader("🖼️ Upload CBCT Images")
img_files = st.file_uploader("Upload CBCT Slices (optional)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Process CSV
if csv_file is not None:
    df = pd.read_csv(csv_file)
    st.dataframe(df.head())

    if st.button("📥 Save to Shared Dataset"):
        uid = str(uuid.uuid4())[:8]
        filename = f"data_{uid}.csv"
        df.to_csv(os.path.join(DATASET_DIR, filename), index=False)
        st.success(f"✅ CSV data saved as {filename}")

# Process image uploads
if img_files:
    for img in img_files:
        image = Image.open(img)
        uid = str(uuid.uuid4())[:8]
        img_path = os.path.join(DATASET_DIR, f"cbct_{uid}.jpg")
        image.save(img_path)
    st.success(f"✅ Uploaded {len(img_files)} images to shared dataset")

# Show current dataset files
st.subheader("📁 Existing Dataset Files")
dataset_files = os.listdir(DATASET_DIR)
for file in dataset_files:
    st.write(f"- {file}")
