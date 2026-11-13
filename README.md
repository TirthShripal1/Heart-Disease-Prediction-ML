# Heart Disease Prediction Using Machine Learning

An educational machine-learning classification project that uses patient-record features from a properly sourced dataset to predict the dataset's heart-disease target label. It includes data validation, preprocessing, model training, evaluation, and single-record prediction.

> **Medical limitation:** This project is for learning and research only. It is not a medical device and must not be used for diagnosis, treatment, screening, or personal medical decisions. Predictions are not clinically validated.

## Repository structure

```text
Heart-Disease-Prediction-ML/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── data/
│   └── README.md
├── models/
│   └── README.md
├── reports/
│   └── README.md
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── train.py
    ├── evaluate.py
    └── predict.py
```

## 1. Installation

Use Python 3.10 or 3.11.

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt and run `.venv\Scripts\activate.bat`.

### macOS / Linux

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Dataset preparation

This project expects a CSV file with one row per record and one target column. The default target column name is `target`, with `1` meaning the positive class and `0` meaning the negative class. Confirm the meaning and coding from the exact dataset documentation; change `--target` if needed.

Place your CSV here:

```text
data/heart.csv
```

The training script creates a stratified train/test split. It automatically handles numeric and categorical columns using median/mode imputation, scaling for numeric features, and one-hot encoding for categorical features.

**Dataset sourcing and attribution:** No dataset is included. You may investigate the UCI Machine Learning Repository's Heart Disease dataset, but check the current dataset documentation, provenance, and terms before use. Record the exact dataset version, source URL, authors/maintainers, access date, citation, target definition, and permitted use in `data/README.md`. Do not include private or identifiable health records.

## 3. Train

From the repository root:

```bash
python -m src.train --data data/heart.csv --target target
```

Optional settings:

```bash
python -m src.train --help
```

The script compares Logistic Regression and Random Forest using a stratified holdout test set and selects the model with the higher test accuracy. For a rigorous scientific comparison, do not use the test set for model selection; use a separate validation set or cross-validation instead. This starter project is intended for learning and reports the limitation explicitly.

The selected fitted pipeline is saved to `models/heart_disease_model.joblib`; metadata is saved to `models/model_metadata.json`.

## 4. Evaluate

```bash
python -m src.evaluate --data data/heart.csv --target target
```

Evaluation trains the same model-selection workflow and prints accuracy, precision, recall, F1-score, ROC-AUC where defined, and a confusion matrix. Metrics are calculated from the supplied dataset split and are not evidence of clinical performance.

## 5. Predict a single record

After training, create a JSON file whose keys match the feature columns used by the dataset. Example structure only (values are placeholders, not medical advice):

```json
{
  "age": 55,
  "sex": 1,
  "cp": 0,
  "trestbps": 130,
  "chol": 240,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 1.2,
  "slope": 1,
  "ca": 0,
  "thal": 2
}
```

Run:

```bash
python app.py --input sample_patient.json
```

Your dataset may use different feature columns. The JSON keys must exactly match the features used during training. Do not use real personal health information in public repositories or demos.

## 6. Responsible use

- Verify the dataset's target definition and feature meanings.
- Keep the test set separate from model development and tuning.
- Prefer patient-level splitting when multiple records per person exist.
- Check for missing values, duplicates, class imbalance, and data leakage.
- Report precision, recall, F1-score, ROC-AUC, and the confusion matrix, not accuracy alone.
- Validate on independent data before making any generalization claim.
- Do not claim clinical usefulness or fabricated performance metrics.

## License

No LICENSE file is included. Before redistribution, choose a project license and verify that the dataset's terms permit your intended use.
