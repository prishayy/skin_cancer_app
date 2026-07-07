
import os
import json
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

from utils.segmentation import load_segmentation_model, segment_image
from utils.handcrafted_features import extract_all_handcrafted_features
from utils.cnn_embeddings import load_embedding_model, extract_efficientnet_embedding
from utils.prediction import load_prediction_artifacts, build_hybrid_vector, predict_class

st.set_page_config(
    page_title="Skin Cancer Classification",
    page_icon="🧬",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

CONFIG_PATH = os.path.join(BASE_DIR, "app_config.json")
CLASS_MAPPING_PATH = os.path.join(BASE_DIR, "class_mapping.json")

with open(CONFIG_PATH, "r") as f:
    APP_CONFIG = json.load(f)

with open(CLASS_MAPPING_PATH, "r") as f:
    CLASS_MAPPING = json.load(f)

st.title("Skin Lesion Classification System")
st.caption("Full pipeline: ResUNet segmentation + handcrafted features + EfficientNet-B0 embeddings + final classifier")

st.warning(
    "This application is for academic/research demonstration only. "
    "It is not a medical diagnostic tool."
)

@st.cache_resource
def load_all_models():
    segmentation_model = load_segmentation_model(
        model_path=os.path.join(MODEL_DIR, "best_resunet.pth"),
        encoder_name=APP_CONFIG.get("segmentation_encoder", "resnet18")
    )

    embedding_model, embedding_transform = load_embedding_model()

    prediction_artifacts = load_prediction_artifacts(
        model_dir=MODEL_DIR,
        classifier_file="best_classifier.pkl",
        label_encoder_file="label_encoder.pkl",
        hybrid_feature_file="hybrid_feature_names.csv",
        selected_feature_file="selected_feature_names.csv"
    )

    return segmentation_model, embedding_model, embedding_transform, prediction_artifacts

with st.spinner("Loading models..."):
    segmentation_model, embedding_model, embedding_transform, prediction_artifacts = load_all_models()

uploaded_file = st.file_uploader(
    "Upload a raw skin lesion image",
    type=["jpg", "jpeg", "png"]
)

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
