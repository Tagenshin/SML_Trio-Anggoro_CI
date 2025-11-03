import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import os
import warnings
import sys
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

if __name__ == "__main__":
    # --- Ambil parameter dari MLproject ---
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--penalty", type=str, default="l2")
    parser.add_argument("--C", type=float, default=0.1)
    parser.add_argument("--solver", type=str, default="liblinear")
    args = parser.parse_args()

    # --- Set MLflow Tracking ---
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("SML_Trio-Anggoro_CI")

    # --- Load data ---
    file_path = (
        sys.argv[3]
        if len(sys.argv) > 3
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), "Telco-Customer-Churn_preprocessing.csv")
    )

    data = pd.read_csv(file_path)
    X = data.drop("Churn", axis=1)
    y = data["Churn"]

    # --- Split data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    input_example = pd.DataFrame(X_train[:5], columns=X.columns)

    with mlflow.start_run():
        mlflow.sklearn.autolog()

        # Inisialisasi model dengan parameter dari MLproject
        model = LogisticRegression(
            penalty=args.penalty,
            C=args.C,
            solver=args.solver,
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Hitung metrik
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Log metrik & model
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            registered_model_name="CreditScoringModel_LogReg",
        )

        print("Logistic Regression model trained & logged to MLflow")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC AUC  : {roc_auc:.4f}")
