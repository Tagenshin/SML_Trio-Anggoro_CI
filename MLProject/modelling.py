import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import numpy as np
import warnings
import os
import sys

warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

# === Fungsi utama ===
def main(dataset_path, penalty, C, solver):
    # Setup MLflow Tracking
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("SML_Trio-Anggoro_CI")

    # Load dataset
    data = pd.read_csv(dataset_path)
    X = data.drop("Churn", axis=1)
    y = data["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    input_example = pd.DataFrame(X_train[:5], columns=X.columns)

    # === Jalankan Run MLflow ===
    active_run = mlflow.active_run()
    if active_run is None:
        run = mlflow.start_run()
    else:
        run = active_run

    mlflow.sklearn.autolog()

    # Model
    model = LogisticRegression(penalty=penalty, C=C, solver=solver)
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Logging manual ke MLflow
    mlflow.log_param("penalty", penalty)
    mlflow.log_param("C", C)
    mlflow.log_param("solver", solver)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    # Simpan model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example,
        registered_model_name="CreditScoringModel_LogReg",
    )

    print("\n Logistic Regression model trained & logged to MLflow")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")

    # Tutup run jika baru dibuat manual
    if active_run is None:
        mlflow.end_run()


# === Entry point ===
if __name__ == "__main__":
    # Jika dijalankan via MLproject
    if len(sys.argv) > 1 and sys.argv[1].endswith(".csv"):
        dataset = sys.argv[1]
        penalty = sys.argv[2] if len(sys.argv) > 2 else "l2"
        C = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
        solver = sys.argv[4] if len(sys.argv) > 4 else "liblinear"
    else:
        # Jika dijalankan manual (python modelling.py --dataset ...)
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, required=True)
        parser.add_argument("--penalty", type=str, default="l2")
        parser.add_argument("--C", type=float, default=0.1)
        parser.add_argument("--solver", type=str, default="liblinear")
        args = parser.parse_args()

        dataset = args.dataset
        penalty = args.penalty
        C = args.C
        solver = args.solver

    # Jalankan fungsi utama
    main(dataset, penalty, C, solver)
