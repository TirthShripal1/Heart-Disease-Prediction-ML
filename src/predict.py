from pathlib import Path
import json
import joblib
import pandas as pd


def predict_record(record, model_path="models/heart_disease_model.joblib",
                   metadata_path="models/model_metadata.json"):
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model not found: {model_path}. Train the model first.")
    if not Path(metadata_path).is_file():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, encoding="utf-8") as file:
        metadata = json.load(file)

    expected_columns = metadata["feature_columns"]
    missing = [column for column in expected_columns if column not in record]
    extra = [column for column in record if column not in expected_columns]
    if missing or extra:
        raise ValueError(
            f"Input keys must match training features. Missing: {missing}; extra: {extra}"
        )

    row = pd.DataFrame([[record[column] for column in expected_columns]],
                       columns=expected_columns)
    model = joblib.load(model_path)
    predicted_label = model.predict(row)[0]

    positive_probability = None
    if hasattr(model, "predict_proba"):
        classes = list(model.named_steps["classifier"].classes_)
        positive_class = metadata["positive_class"]
        if positive_class in classes:
            positive_probability = float(
                model.predict_proba(row)[0][classes.index(positive_class)]
            )

    return predicted_label.item() if hasattr(predicted_label, "item") else predicted_label, positive_probability
