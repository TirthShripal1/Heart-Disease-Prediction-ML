import argparse
import json
from src.predict import predict_record


def main():
    parser = argparse.ArgumentParser(
        description="Predict a dataset label for one record (educational use only)."
    )
    parser.add_argument("--input", required=True, help="Path to a JSON record")
    parser.add_argument("--model", default="models/heart_disease_model.joblib")
    parser.add_argument("--metadata", default="models/model_metadata.json")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as file:
        record = json.load(file)

    label, probability = predict_record(record, args.model, args.metadata)
    print(f"Predicted dataset class: {label}")
    if probability is not None:
        print(f"Model score for positive class: {probability:.4f}")
    print("Educational output only. This is not a medical diagnosis.")


if __name__ == "__main__":
    main()
