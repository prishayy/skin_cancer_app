
import os
import joblib
import numpy as np
import pandas as pd

def load_prediction_artifacts(
    model_dir,
    classifier_file,
    label_encoder_file,
    hybrid_feature_file,
    selected_feature_file
):
    classifier = joblib.load(os.path.join(model_dir, classifier_file))
    label_encoder = joblib.load(os.path.join(model_dir, label_encoder_file))

    hybrid_features = pd.read_csv(os.path.join(model_dir, hybrid_feature_file))["feature_name"].tolist()

    selected_df = pd.read_csv(os.path.join(model_dir, selected_feature_file))

    if "selected_feature_name" in selected_df.columns:
        selected_features = selected_df["selected_feature_name"].tolist()
    elif "feature_name" in selected_df.columns:
        selected_features = selected_df["feature_name"].tolist()
    else:
        selected_features = selected_df.iloc[:, -1].tolist()

    return {
        "classifier": classifier,
        "label_encoder": label_encoder,
        "hybrid_features": hybrid_features,
        "selected_features": selected_features
    }

def build_hybrid_vector(handcrafted_features, embedding_features, prediction_artifacts):
    all_features = {}
    all_features.update(handcrafted_features)
    all_features.update(embedding_features)

    feature_names = prediction_artifacts["hybrid_features"]

    vector = []
    missing = []

    for name in feature_names:
        if name in all_features:
            vector.append(all_features[name])
        else:
            vector.append(0.0)
            missing.append(name)

    vector = np.array(vector, dtype=np.float32).reshape(1, -1)

    return vector

def predict_class(hybrid_vector, prediction_artifacts):
    classifier = prediction_artifacts["classifier"]
    label_encoder = prediction_artifacts["label_encoder"]

    pred_idx = classifier.predict(hybrid_vector)[0]
    pred_class = label_encoder.inverse_transform([pred_idx])[0]

    probabilities = {}

    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(hybrid_vector)[0]
        classes = label_encoder.inverse_transform(np.arange(len(proba)))

        for cls, p in zip(classes, proba):
            probabilities[cls] = float(p)

        confidence = float(np.max(proba))
    else:
        probabilities[pred_class] = 1.0
        confidence = 1.0

    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "probabilities": probabilities
    }
