from pathlib import Path
import pandas as pd


def load_data(csv_path, target_column):
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. See data/README.md for setup."
        )

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("The CSV file contains no rows.")
    if target_column not in frame.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(frame.columns)}"
        )
    if frame[target_column].isna().any():
        raise ValueError("Target column contains missing values; clean labels first.")

    unique_values = sorted(frame[target_column].unique().tolist())
    if len(unique_values) != 2:
        raise ValueError(
            f"Expected a binary target with two values; found {unique_values}"
        )
    return frame
