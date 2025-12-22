import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# === LOAD DATA ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "HR-Employee-Attrition_preprosessing")

train_df = pd.read_csv(os.path.join(DATA_DIR, "train_processed.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test_processed.csv"))

X_train = train_df.drop("Attrition", axis=1)
y_train = train_df["Attrition"]
X_test  = test_df.drop("Attrition", axis=1)
y_test  = test_df["Attrition"]

# === MLFLOW CONFIG ===
mlflow.set_experiment("HR_Attrition_CI")

with mlflow.start_run() as run:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, artifact_path="model")

    # === SIMPAN RUN_ID UNTUK DOCKER STEP ===
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)
