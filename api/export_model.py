"""Export a registered churn-rf version as the pickle the Flask API serves (Q6).

The registry stores each model in MLflow's skops format, while api/app.py loads a
plain pickle. This loads the version with MLflow and pickles the scikit-learn
pipeline, so the API answers exactly as that version does on the Azure endpoint.

    export MLFLOW_TRACKING_URI=$(.venv/bin/python pipeline/workspace.py)
    .venv/bin/python api/export_model.py --version 2      # -> outputs/model/model.pkl
"""
import argparse
import pickle
import sys
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from register import MODEL_NAME  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="2")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "model" / "model.pkl"))
    args = parser.parse_args()

    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{args.version}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved {MODEL_NAME} version {args.version} to {output}")


if __name__ == "__main__":
    main()
