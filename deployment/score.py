"""Scoring script for the online endpoint deployments.

Accepts the same request body as Azure's no-code MLflow deployments
({"input_data": {"columns": [...], "data": [[...]]}}), and also returns
the churn probability for each row.
"""
import json
import os
from pathlib import Path

import mlflow
import pandas as pd

model = None


def init():
    global model
    # The registered MLflow model sits somewhere under AZUREML_MODEL_DIR.
    model_dir = next(Path(os.environ["AZUREML_MODEL_DIR"]).rglob("MLmodel")).parent
    # The sklearn flavor (not pyfunc) exposes predict_proba; the model's bundled
    # features.py is added to sys.path by MLflow when it loads.
    model = mlflow.sklearn.load_model(str(model_dir))


def run(raw_data):
    payload = json.loads(raw_data)["input_data"]
    if isinstance(payload, dict):
        df = pd.DataFrame(payload["data"], columns=payload["columns"])
    else:
        df = pd.DataFrame(payload)  # list of {column: value} records
    df = df.astype(float)

    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]
    return [
        {"prediction": int(p), "churn_probability": round(float(q), 4)}
        for p, q in zip(predictions, probabilities)
    ]
