import mlflow
import mlflow.data
import mlflow.sklearn
import dagshub
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import dagshub
dagshub.init(repo_owner='DhimaanDutta07', repo_name='my-first-repo', mlflow=True)

mlflow.set_experiment("GridSearch_MultiModel_Iris")

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["lbfgs"]
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
for model_name, config in models.items():
    with mlflow.start_run(run_name=model_name):
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


        mlflow.autolog()
        
        mlflow.set_tag("V.1.0")

        mlflow.sklearn.log_model(
            sk_model=best_model)
        
        