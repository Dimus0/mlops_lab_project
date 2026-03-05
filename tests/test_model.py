import json
import os
import pickle
import pandas as pd

def test_data_schema():
    data_path = os.getenv("DATA_PATH", "data/prepared/train.csv")
    assert os.path.exists(data_path), f"Data file {data_path} does not exist"

    df = pd.read_csv(data_path)
    required_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges', 'Churn']

    missing_columns = [col for col in required_columns if col not in df.columns]
    assert not missing_columns, f"Missing columns: {missing_columns}"

    assert df['Churn'].isin([1, 0]).all(), "Target column 'Churn' contains invalid values"

def test_quality_of_model():
    metrics_path = os.path.join("models/metrics", "metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert metrics["f1_test"] > 0.78, "F1 score on test set is too low"

def test_artifact_exists():
    model_path = os.path.join("models", "best_model.pkl")
    assert os.path.exists(model_path), "Model artifact does not exist"
    assert os.path.exists("models/metrics/metrics.json"), "Metrics file does not exist"
