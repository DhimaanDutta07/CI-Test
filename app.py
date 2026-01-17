import mlflow
import mlflow.sklearn
import mlflow.data
import dagshub
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ----------------------------------
# DAGsHub + MLflow Init
# ----------------------------------
dagshub.init(
    repo_owner="DhimaanDutta07",
    repo_name="my-first-repo",
    mlflow=True
)

mlflow.set_experiment("GridSearch_MultiModel_Iris")

# ----------------------------------
# Enable autolog (ONCE, BEFORE training)
# ----------------------------------
mlflow.sklearn.autolog()

# ----------------------------------
# Load data (as DataFrame for lineage)
# ----------------------------------
X, y = load_iris(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Log datasets
train_dataset = mlflow.data.from_pandas(X_train, name="X_train")
test_dataset = mlflow.data.from_pandas(X_test, name="X_test")

# ----------------------------------
# Models & Params
# ----------------------------------
models = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "C": [0.01, 0.1, 1, 10]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5, None]
        }
    },
    "SVC": {
        "model": SVC(),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        }
    }
}

# ----------------------------------
# Training Loop
# ----------------------------------
for model_name, config in models.items():

    with mlflow.start_run(run_name=model_name):

        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(test_dataset, context="testing")

        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Manual metrics (extra clarity)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("cv_best_score", grid.best_score_)

        # Tags
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("version", "v1.0")

        # Log & register model
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=f"Iris_{model_name}"
        )

        print(f"✅ {model_name} logged successfully")
