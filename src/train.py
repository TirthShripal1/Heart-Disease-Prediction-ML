import argparse
import json
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data_loader import load_data
from .preprocessing import build_preprocessor


def train_model(data_path, target_column, test_size=0.2, random_state=42):
    frame = load_data(data_path, target_column)
    features = frame.drop(columns=[target_column])
    target = frame[target_column]

    if features.shape[1] == 0:
        raise ValueError("The dataset must contain at least one feature column.")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced",
        ),
    }

    results = {}
    fitted_models = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocessing", build_preprocessor(X_train)),
            ("classifier", estimator),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        results[name] = float(accuracy_score(y_test, predictions))
        fitted_models[name] = pipeline

    selected_name = max(results, key=results.get)
    selected_model = fitted_models[selected_name]

    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    joblib.dump(selected_model, output_dir / "heart_disease_model.joblib")

    metadata = {
        "target_column": target_column,
        "feature_columns": features.columns.tolist(),
        "class_labels": [value.item() if hasattr(value, "item") else value
                         for value in sorted(target.unique().tolist())],
        "positive_class": 1 if 1 in target.unique() else sorted(target.unique().tolist())[-1],
        "selected_model": selected_name,
        "holdout_accuracy": results[selected_name],
        "candidate_accuracies": results,
        "test_size": test_size,
        "random_state": random_state,
        "selection_note": (
            "For this educational starter, candidates are compared on the holdout "
            "test split. For unbiased final reporting, select models using a "
            "validation split or cross-validation, then evaluate once on untouched test data."
        ),
    }
    with open(output_dir / "model_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return selected_model, metadata, X_test, y_test


def main():
    parser = argparse.ArgumentParser(description="Train heart-disease ML models.")
    parser.add_argument("--data", default="data/heart.csv")
    parser.add_argument("--target", default="target")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.1 <= args.test_size <= 0.5:
        parser.error("--test-size must be between 0.1 and 0.5")

    _, metadata, _, _ = train_model(
        args.data, args.target, args.test_size, args.seed
    )
    print(f"Selected model: {metadata['selected_model']}")
    print(f"Holdout accuracy: {metadata['holdout_accuracy']:.4f}")
    print("Saved model and metadata in models/")


if __name__ == "__main__":
    main()
