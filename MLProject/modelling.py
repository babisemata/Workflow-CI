import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train_df = pd.read_csv("HR-Employee-Attrition_preprosessing/train_processed.csv")
test_df = pd.read_csv("HR-Employee-Attrition_preprosessing/test_processed.csv")

X_train = train_df.drop(columns=["Attrition"])
y_train = train_df["Attrition"]
X_test = test_df.drop(columns=["Attrition"])
y_test = test_df["Attrition"]

mlflow.set_experiment("HR_Attrition_CI")

with mlflow.start_run():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, "model")

with open("run_id.txt", "w") as f:
    f.write(mlflow.active_run().info.run_id)

