import os
import json
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
from PIL import Image
import torch
import gdown

# ============================================================
# GOOGLE DRIVE FILE IDs
# ============================================================
RESUNET_FILE_ID = "1m7oQpHw8ter47_1ysXNyImE50mr94l5V"
CLASSIFIER_FILE_ID = "1nDtTTLKj1mRwIOA-5tbVB18ECY5HdVaW"
ENCODER_FILE_ID = "Y17BLuhzPCloQqBX5r7c96vmIZy1bHwcHx"

# ============================================================
# IMPORTS
# ============================================================
from utils.segmentation import load_segmentation_model, segment_image
from utils.handcrafted_features import extract_all_handcrafted_features
from utils.cnn_embeddings import load_embedding_model, extract_efficientnet_embedding
from utils.prediction import load_prediction_artifacts, build_hybrid_vector, predict_class

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Skin Cancer Classification",
    page_icon="🧬",
    layout="wide"
)

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

CONFIG_PATH = os.path.join(BASE_DIR, "app_config.json")
CLASS_MAPPING_PATH = os.path.join(BASE_DIR, "class_mapping.json")

# ============================================================
# LOAD CONFIG FILES
# ============================================================
with open(CONFIG_PATH, "r") as f:
    APP_CONFIG = json.load(f)

with open(CLASS_MAPPING_PATH, "r") as f:
    CLASS_MAPPING = json.load(f)

# ============================================================
# APP UI
# ============================================================
st.title("Skin Lesion Classification System")
st.caption("Full pipeline: ResUNet segmentation + handcrafted features + EfficientNet-B0 embeddings + final classifier")

st.warning(
    "This application is for academic/research demonstration only. "
    "It is not a medical diagnostic tool."
)

# ============================================================
# FUNCTION TO DOWNLOAD AND LOAD MODELS
# ============================================================
@st.cache_resource
def download_and_load_models():
    """
    Downloads models from Google Drive if they don't exist locally,
    then loads and returns them.
    This runs only ONCE per deployment (cached).
    """
    # Create models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # File paths
    resunet_path = os.path.join(MODEL_DIR, "best_resunet.pth")
    classifier_path = os.path.join(MODEL_DIR, "best_classifier.pkl")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    hybrid_feature_path = os.path.join(MODEL_DIR, "hybrid_feature_names.csv")
    selected_feature_path = os.path.join(MODEL_DIR, "selected_feature_names.csv")

    # ============================================================
    # DOWNLOAD RESUNET MODEL (54 MB)
    # ============================================================
    if not os.path.exists(resunet_path):
        st.info("📥 Downloading ResUNet model (54 MB)... This may take 1-2 minutes.")
        try:
            gdown.download(
                f'https://drive.google.com/uc?id={RESUNET_FILE_ID}',
                resunet_path,
                quiet=False
            )
            st.success("✅ ResUNet model downloaded!")
        except Exception as e:
            st.error(f"❌ Failed to download ResUNet model: {e}")
            st.stop()

    # ============================================================
    # DOWNLOAD CLASSIFIER
    # ============================================================
    if not os.path.exists(classifier_path):
        st.info("📥 Downloading classifier model...")
        try:
            gdown.download(
                f'https://drive.google.com/uc?id={CLASSIFIER_FILE_ID}',
                classifier_path,
                quiet=False
            )
            st.success("✅ Classifier downloaded!")
        except Exception as e:
            st.error(f"❌ Failed to download classifier: {e}")
            st.stop()

    # ============================================================
    # DOWNLOAD LABEL ENCODER
    # ============================================================
    if not os.path.exists(encoder_path):
        st.info("📥 Downloading label encoder...")
        try:
            gdown.download(
                f'https://drive.google.com/uc?id={ENCODER_FILE_ID}',
                encoder_path,
                quiet=False
            )
            st.success("✅ Label encoder downloaded!")
        except Exception as e:
            st.error(f"❌ Failed to download label encoder: {e}")
            st.stop()

    # ============================================================
    # NOW LOAD THE MODELS
    # ============================================================
    # 1. Load ResUNet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    segmentation_model = load_segmentation_model(
        model_path=resunet_path,
        encoder_name=APP_CONFIG.get("segmentation_encoder", "resnet18")
    )

    # 2. Load EfficientNet embedding model
    embedding_model, embedding_transform = load_embedding_model()

    # 3. Load prediction artifacts
    prediction_artifacts = load_prediction_artifacts(
        model_dir=MODEL_DIR,
        classifier_file="best_classifier.pkl",
        label_encoder_file="label_encoder.pkl",
        hybrid_feature_file="hybrid_feature_names.csv",
        selected_feature_file="selected_feature_names.csv"
    )

    return segmentation_model, embedding_model, embedding_transform, prediction_artifacts

# ============================================================
# LOAD MODELS (with spinner)
# ============================================================
with st.spinner("Loading models..."):
    segmentation_model, embedding_model, embedding_transform, prediction_artifacts = download_and_load_models()

# ============================================================
# FILE UPLOADER
# ============================================================
uploaded_file = st.file_uploader(
    "Upload a raw skin lesion image",
    type=["jpg", "jpeg", "png"]
)

# ============================================================
# PREDICTION PIPELINE
# ============================================================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Input Image")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Run Full Prediction Pipeline"):
        with st.spinner("Running segmentation..."):
            segmented_image, mask_image = segment_image(
                pil_image=image,
                model=segmentation_model,
                image_size=APP_CONFIG.get("image_size", 256),
                threshold=APP_CONFIG.get("mask_threshold", 0.5)
            )

        # Show results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Original", use_container_width=True)
        with col2:
            st.image(mask_image, caption="Predicted Mask", use_container_width=True)
        with col3:
            st.image(segmented_image, caption="Segmented Lesion", use_container_width=True)

        with st.spinner("Extracting handcrafted features..."):
            handcrafted_features = extract_all_handcrafted_features(segmented_image)

        with st.spinner("Extracting EfficientNet-B0 embedding..."):
            embedding_features = extract_efficientnet_embedding(
                segmented_image,
                embedding_model,
                embedding_transform
            )

        with st.spinner("Building hybrid feature vector and predicting..."):
            hybrid_vector = build_hybrid_vector(
                handcrafted_features=handcrafted_features,
                embedding_features=embedding_features,
                prediction_artifacts=prediction_artifacts
            )

            prediction = predict_class(
                hybrid_vector=hybrid_vector,
                prediction_artifacts=prediction_artifacts
            )

        pred_class = prediction["predicted_class"]
        confidence = prediction["confidence"]
        probabilities = prediction["probabilities"]

        st.subheader("Prediction Result")

        readable_name = CLASS_MAPPING.get(pred_class, pred_class)

        st.metric("Predicted Class", pred_class)
        st.write("**Description:**", readable_name)
        st.metric("Confidence", f"{confidence * 100:.2f}%")

        prob_df = pd.DataFrame({
            "Class": list(probabilities.keys()),
            "Probability": list(probabilities.values())
        }).sort_values("Probability", ascending=False)

        st.subheader("Class Probabilities")
        st.dataframe(prob_df, use_container_width=True)

        st.bar_chart(prob_df.set_index("Class"))

        st.subheader("Feature Vector Check")
        st.write("Hybrid vector shape:", hybrid_vector.shape)
        st.write("Handcrafted features extracted:", len(handcrafted_features))
        st.write("EfficientNet features extracted:", len(embedding_features))
else:
    st.info("Upload an image to begin.")
