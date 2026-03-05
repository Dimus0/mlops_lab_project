import os

import joblib
import mlflow
import argparse
import numpy as np
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, root_mean_squared_error,f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import json

"""
    How to run:
    python src/train.py --n_estimators 100 --criterion entropy --max_depth 12
"""

def parser_args():
    parser = argparse.ArgumentParser(description="Train a model on the Telco Customer Churn dataset.")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the random forest.")
    parser.add_argument("--criterion", type=str, default="gini", help="Number of trees in the random forest.")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of the tree.")

    return parser.parse_args()

def main():
    args = parser_args()

    train_path = os.path.join("data", "prepared", "train.csv")
    test_path = os.path.join("data", "prepared", "test.csv")


    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print("Data loaded successfully.")

    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]
    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    feature_names = X_train.columns.tolist()

    mlflow.set_tracking_uri(r"sqlite:///D:/Python/MLOPS/mlops_lab_1/mlflow.db")
    mlflow.set_experiment("MLOPS_LAB_1-TelcoChurn")

    with mlflow.start_run():

        mlflow.set_tags({
            "author": "Puhachevskyi Dmytro",
            "dataset": "Telco Customer Churn",
            "model": "RandomForestClassifier"
        })

        mlflow.log_params({
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            'criterion': args.criterion,
        })

        print(f"Training the model...")

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            criterion=args.criterion,
            class_weight="balanced",
            random_state=42
        )

        model.fit(X_train,y_train)

        # predict
        y_test_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)


        acc_test = accuracy_score(y_test,y_test_pred)
        f1_test = f1_score(y_test,y_test_pred)

        acc_train = accuracy_score(y_train,y_train_pred)
        f1_train = f1_score(y_train,y_train_pred)

        mlflow.log_metrics({
            "accuracy_train": acc_train,
            "f1_train": f1_train,
            "accuracy_test": acc_test,
            "f1_test": f1_test
        })

        print(
        f"Train | Accuracy: {acc_train:.4f}, F1: {f1_train:.4f}\n"
        f"Test  | Accuracy: {acc_test:.4f}, F1: {f1_test:.4f}"
        )

        metrics_path = os.path.join("models", "metrics", "metrics.json")

        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        print(f"Saving metrics to {metrics_path}...")

        with open(metrics_path, "w") as f:
            json.dump({
                "accuracy_train": acc_train,
                "f1_train": f1_train,
                "accuracy_test": acc_test,
                "f1_test": f1_test
            }, f)


        model_output_path = os.path.join("models", "best_model.pkl")
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(model, model_output_path)
        print(f"Model weights saved to {model_output_path}")

        '''
            Plot feature importance
        '''
        feature_importances = model.feature_importances_
        indices = np.argsort(feature_importances)[::-1]

        fig_importance, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_title("Feature Importances")
        ax1.bar(range(X_train.shape[1]), feature_importances[indices], align="center")
        ax1.set_xticks(range(X_train.shape[1]))
        ax1.set_xticklabels([feature_names[i] for i in indices], rotation=45)
        fig_importance.tight_layout()

        mlflow.log_figure(fig_importance, "plots/feature_importance.png")
        plt.close(fig_importance)

        '''
            Plot confusion matrix
        '''
        fig_cm, ax2 = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, ax=ax2)
        ax2.set_title("Confusion Matrix")

        fig_cm.savefig("confusion_matrix.png")

        mlflow.log_figure(fig_cm, "plots/confusion_matrix.png")
        plt.close(fig_cm)


        mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest_classifier",
            registered_model_name="RandomForestClassifier_TelcoChurn",
            serialization_format="skops" 
        )
        print("Run complete. Artifacts logged.")

if __name__ == "__main__":
    main()

