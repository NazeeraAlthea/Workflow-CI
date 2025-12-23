import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os

mlflow.set_experiment("Student-Performance")

# Load dataset hasil preprocessing
df = pd.read_csv(
    "student-performance_preprocessing/data.csv"
)

X = df.drop(columns=["Performance_Level"])
y = df["Performance_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_grid = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "solver": ["lbfgs"],
    "max_iter": [500, 1000]
}

base_model = LogisticRegression()

grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)


grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro")
rec = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

mlflow.log_params(grid_search.best_params_)
mlflow.log_metric("accuracy", acc)
mlflow.log_metric("precision", prec)
mlflow.log_metric("recall", rec)
mlflow.log_metric("f1_score", f1)

mlflow.sklearn.log_model(best_model, "model")

print("Best Params:", grid_search.best_params_)
print("Accuracy:", acc)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")

cm_path = "confusion_matrix.png"
plt.savefig(cm_path)
plt.close()

mlflow.log_artifact(cm_path)

# Classification Report
report = classification_report(y_test, y_pred)

report_path = "classification_report.txt"
with open(report_path, "w") as f:
    f.write(report)

mlflow.log_artifact(report_path)
