"""Connect to the Azure ML workspace.

Run directly to print the workspace's MLflow tracking URI, so the src/ scripts
log to the AzureML experiment instead of the local mlflow.db:

    export MLFLOW_TRACKING_URI=$(.venv/bin/python pipeline/workspace.py)

Uses config.json (downloadable from the workspace page in Azure ML Studio) in the
repo root, or the AZURE_SUBSCRIPTION_ID / AZURE_RESOURCE_GROUP / AZUREML_WORKSPACE_NAME
environment variables. Authentication comes from `az login`.
"""
import contextlib
import os
import sys
from pathlib import Path

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ROOT = Path(__file__).resolve().parent.parent


def get_ml_client():
    credential = DefaultAzureCredential()
    if "AZURE_SUBSCRIPTION_ID" in os.environ:
        return MLClient(
            credential,
            subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
            resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
            workspace_name=os.environ["AZUREML_WORKSPACE_NAME"],
        )
    return MLClient.from_config(credential, path=str(ROOT / "config.json"))


if __name__ == "__main__":
    # Keep SDK chatter off stdout so only the URI is captured by $(...).
    with contextlib.redirect_stdout(sys.stderr):
        ml_client = get_ml_client()
        uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    print(uri)
