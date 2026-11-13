import argparse
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from .train import train_model


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate candidate models.")
    parser.add_argument("--data", default="data/heart.csv")
    parser.add_argument("--target", default="target")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model, metadata, X_test, y_test = train_model(
        args.data, args.target, args.test_size, args.seed
    )
    predictions = model.predict(X_test)

    print(f"Selected model: {metadata['selected_model']}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
        probabilities = model.predict_proba(X_test)[:, 1]
        try:
            print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
        except ValueError:
            print("ROC-AUC could not be calculated for this test split.")

    print("\nThese are dataset-split metrics, not clinical performance claims.")


if __name__ == "__main__":
    main()
