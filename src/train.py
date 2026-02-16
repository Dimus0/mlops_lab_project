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

def preprocessing_and_split_X_y(df):

    df = df.drop(columns=["customerID"])
    df.dropna(inplace=True)

    encoder = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == "str" or df[column].dtype == "object":
            df[column] = encoder.fit_transform(df[column])
    
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    smote = SMOTE(random_state=42)
    X_balance, y_balanced = smote.fit_resample(X, y)


    return df, X_balance, y_balanced

def main():
    args = parser_args()
    
    df = pd.read_csv(r"D:\Python\MLOPS\mlops_lab_1\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv",sep=",")

    df, X, y = preprocessing_and_split_X_y(df)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    mlflow.set_tracking_uri(r"file:///D:/Python/MLOPS/mlops_lab_1/mlruns")
    mlflow.set_experiment("MLOPS_LAB_1-TelcoChurn")

    with mlflow.start_run():

        mlflow.set_tag({
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
            class_weight="balanced",
            random_state=42
        )

        model.fit(X_train,y_train)

        # predict
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)


        acc_test = accuracy_score(y_test,y_pred)
        f1_test = f1_score(y_test,y_pred)

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

        '''
            Plot feature importance
        '''
        feature_importances = model.feature_importances_
        indices = np.argsort(feature_importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(X.shape[1]), feature_importances[indices], align="center")
        plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()

        plt_path_feature_importance = "feature_importance.png"
        plt.savefig(plt_path_feature_importance)
        plt.close()

        mlflow.log_artifact(plt_path_feature_importance)


        '''
            Plot confusion matrix
        '''
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        plt.title("Confusion Matrix")

        plot_path = "confusion_matrix.png"
        plt.savefig(plot_path)
        plt.close()

        mlflow.log_artifact(plot_path)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest_classifier",
            registered_model_name="RandomForestClassifier_TelcoChurn"
        )
        print("Run complete. Artifacts logged.")

if __name__ == "__main__":
    main()

