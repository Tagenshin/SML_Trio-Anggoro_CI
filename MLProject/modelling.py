import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import warnings
import os
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Ambil file dataset dari argumen atau default ke "train.csv"
    file_path = (
        sys.argv[4]
        if len(sys.argv) > 4
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), "Telco-Customer-Churn_preprocessing.csv")
    )
    df = pd.read_csv(file_path)

    # Pisahkan fitur dan target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Ambil parameter dari argumen (default jika tidak diberikan)
    penalty = sys.argv[1] if len(sys.argv) > 1 else "l2"
    C = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    solver = sys.argv[3] if len(sys.argv) > 3 else "liblinear"

    # Contoh input untuk logging
    input_example = X_train.head(5)

    # # === MLflow Logging ===
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # mlflow.set_experiment("SML_Trio-Anggoro_CI")

    # Jalankan eksperimen di MLflow
    with mlflow.start_run():
        # Buat dan latih model
        model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=1000
        )
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Hitung metrik evaluasi
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)

        # Cetak hasil ke terminal
        print("\n=== Evaluation Results ===")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC AUC  : {roc:.4f}")

        # Log parameter dan metrik ke MLflow
        mlflow.log_param("penalty", penalty)
        mlflow.log_param("C", C)
        mlflow.log_param("solver", solver)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc)

        # Simpan model ke MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )
