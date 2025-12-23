import os
import json
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ===============================
# Load Data
# ===============================
DATA_DIR = "HR-Employee-Attrition_preprosessing"

train_df = pd.read_csv(os.path.join(DATA_DIR, "train_processed.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test_processed.csv"))

X_train = train_df.drop("Attrition", axis=1)
y_train = train_df["Attrition"]
X_test  = test_df.drop("Attrition", axis=1)
y_test  = test_df["Attrition"]

# ===============================
# Simple Tuning
# ===============================
C_values = [0.1, 1.0, 5.0]
solvers = ["liblinear", "lbfgs"]

best_f1 = -1
best_run_id = None

for C in C_values:
    for solver in solvers:
        with mlflow.start_run(run_name=f"ci_logreg_C{C}_{solver}"):

            model = LogisticRegression(
                C=C,
                solver=solver,
                max_iter=1000
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds, zero_division=0),
                "recall": recall_score(y_test, preds, zero_division=0),
                "f1_score": f1_score(y_test, preds, zero_division=0),
            }

            mlflow.log_param("C", C)
            mlflow.log_param("solver", solver)
            mlflow.log_metrics(metrics)

            # ===============================
            # Log Model
            # ===============================
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name="HR_Attrition_LogReg_Model"
            )


            # ===============================
            # Artifact Tambahan
            # ===============================
            # Confusion Matrix
            plt.figure(figsize=(5,4))
            sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d", cmap="Blues")
            plt.tight_layout()
            plt.savefig("confusion_matrix.png")
            plt.close()
            mlflow.log_artifact("confusion_matrix.png")

            # Classification Report
            with open("classification_report.txt", "w") as f:
                f.write(classification_report(y_test, preds, zero_division=0))
            mlflow.log_artifact("classification_report.txt")

            # Metric Summary JSON
            with open("metric_summary.json", "w") as f:
                json.dump(metrics, f, indent=2)
            mlflow.log_artifact("metric_summary.json")

            # ===============================
            # Simpan RUN ID
            # ===============================
            run_id = mlflow.active_run().info.run_id
            with open("run_id.txt", "w") as f:
                f.write(run_id)

            mlflow.log_artifact("run_id.txt")

            print("Active run_id:", run_id)

            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_run_id = run_id

print("BEST RUN ID:", best_run_id)
print("BEST F1:", best_f1)
