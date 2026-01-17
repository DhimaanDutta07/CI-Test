import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize DAGsHub + MLflow
dagshub.init(
    repo_owner="DhimaanDutta07",
    repo_name="my-first-repo",
    mlflow=True
)

mlflow.set_experiment("CI_Test_Iris")
mlflow.sklearn.autolog()  # enable autologging

def main():
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    with mlflow.start_run(run_name="RandomForest"):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Model Accuracy: {acc}")

        # simple quality gate
        assert acc > 0.8, "Accuracy below threshold"

if __name__ == "__main__":
    main()
