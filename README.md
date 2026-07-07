# Skin Cancer Classification Web App

This Streamlit app runs the full skin lesion classification pipeline.

## Pipeline

1. Upload raw skin lesion image
2. Segment lesion using ResUNet
3. Extract handcrafted selected features
4. Extract EfficientNet-B0 image embedding
5. Build hybrid feature vector
6. Predict class using the final trained classifier

## Run Locally

Install dependencies:

pip install -r requirements.txt

Run the app:

streamlit run app.py

## Required Model Files

Place these files inside the models folder:

- best_resunet.pth
- best_classifier.pkl
- label_encoder.pkl
- hybrid_feature_names.csv
- selected_feature_names.csv

## Disclaimer

This app is for academic and research demonstration only.
It is not a medical diagnostic tool.