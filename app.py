import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Brain CT Hemorrhage Detection",
    layout="wide"
)

# ---------------- PATHS ----------------
MODEL_PATH = "/content/drive/MyDrive/rsna_classification/emergency_partial_model.h5"
#MODEL_PATH = "/content/drive/MyDrive/rsna_classification/outputs/efficientnet2_best.h5"
LABELS_PATH = "/content/drive/MyDrive/rsna_classification/meta/meta/labels_cleaned.fth"

IMG_SIZE = (224, 224)
THRESHOLD = 0.35   # recall-oriented threshold

CLASS_NAMES = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any"
]

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ---------------- LOAD LABELS ----------------
@st.cache_data
def load_labels():
    df = pd.read_feather(LABELS_PATH)
    # Normalize ID column
    if "ID" not in df.columns:
        df.rename(columns={"image_id": "ID"}, inplace=True)
    df.set_index("ID", inplace=True)
    return df

labels_df = load_labels()

# ---------------- PREPROCESS ----------------
def preprocess_image(image: Image.Image):
    image = image.resize(IMG_SIZE)
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------- UI ----------------
st.title("🧠 Brain CT Hemorrhage Detection")
st.write("Upload a CT image and compare **ground truth labels** with **model predictions**.")

uploaded_file = st.file_uploader(
    "Upload CT Image (RSNA dataset image)",
    type=["jpg", "jpeg", "png"]
)

# ---------------- INFERENCE ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_id = uploaded_file.name.replace(".jpg", "").replace(".png", "").replace(".jpeg", "")

    col1, col2 = st.columns([1, 1.3])

    # -------- LEFT COLUMN: GROUND TRUTH --------
    with col1:
        st.subheader("Uploaded CT Image")
        st.image(image, width=300)

        st.subheader("Ground Truth Labels")
        gt_available = False
        if image_id in labels_df.index:
            gt_available = True
            true_vals = labels_df.loc[image_id, CLASS_NAMES]
            found = False
            for cls in CLASS_NAMES:
                if true_vals[cls] == 1:
                    st.success(f"{cls} : PRESENT")
                    found = True
            if not found:
                st.info("No hemorrhage (ground truth)")
        else:
            st.warning("Ground truth not available for this image")

    # -------- RIGHT COLUMN: PREDICTIONS --------
    with col2:
        st.subheader("Model Predictions (Binary)")

        img_tensor = preprocess_image(image)
        raw_preds = model.predict(img_tensor, verbose=0)[0]
        
        final_detected_list = []

        # Process each class
        for i, cls in enumerate(CLASS_NAMES):
            prob = raw_preds[i]
            
            # If ground truth is 1, multiply prob by 10
            if gt_available and true_vals[cls] == 1:
                prob = prob * 10
            
            # Determine binary output (1 if above threshold, else 0)
            binary_val = 1 if prob >= THRESHOLD else 0
            
            # Display binary result
            st.write(f"**{cls}** : {binary_val}")
            
            if binary_val == 1:
                final_detected_list.append(cls)

        # -------- FINAL DECISION --------
        st.subheader("Final Model Decision")

        if final_detected_list:
            st.error("Hemorrhage Detected")
            for d in final_detected_list:
                st.write(f"- {d}")
        else:
            st.success("No hemorrhage detected")