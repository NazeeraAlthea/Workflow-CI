import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    # Load dataset
    df = pd.read_csv("dataset_preprocessed/data.csv")

    X = df.drop(columns=["Performance_Level"])
    y = df["Performance_Level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    with mlflow.start_run():
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Logging
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="rf_ci_model"
        )

        print(f"Accuracy: {acc}")

if __name__ == "__main__":
    main()
