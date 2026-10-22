"""Create the telecom-churn data asset (v1 and v2) and show the row count of each version.

Each version is also copied into a landing folder on the workspace datastore,
which the pipeline reads, so a scheduled run always trains on the newest data.
"""
import tempfile
from pathlib import Path

import pandas as pd
from azure.ai.ml._artifacts._artifact_utilities import download_artifact_from_aml_uri
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient

from workspace import ROOT, get_ml_client

DATA_ASSET = "telecom-churn"
VERSIONS = {"1": "telecom_churn_v1.csv", "2": "telecom_churn_v2.csv"}
DATASTORE = "workspaceblobstore"
LANDING_FOLDER = f"{DATA_ASSET}/landing"
LANDING_URI = f"azureml://datastores/{DATASTORE}/paths/{LANDING_FOLDER}/"


def create_versions(ml_client):
    for version, filename in VERSIONS.items():
        try:
            ml_client.data.get(DATA_ASSET, version=version)
            print(f"{DATA_ASSET}:{version} already exists")
        except ResourceNotFoundError:
            ml_client.data.create_or_update(Data(
                name=DATA_ASSET,
                version=version,
                type=AssetTypes.URI_FILE,
                path=str(ROOT / filename),
                description=f"Telecom customer churn data ({filename})",
            ))
            print(f"Created {DATA_ASSET}:{version} from {filename}")
        land_file(ml_client, version, filename)


def land_file(ml_client, version, filename):
    """Copy a data version into the landing folder, named so newer versions sort last."""
    store = ml_client.datastores.get(DATASTORE, include_secrets=True)
    # The workspace may return either an account key or a SAS token for its datastore.
    secret = getattr(store.credentials, "account_key", None) or store.credentials.sas_token
    service = BlobServiceClient(f"https://{store.account_name}.blob.{store.endpoint}", credential=secret)
    name = f"{LANDING_FOLDER}/{DATA_ASSET}-v{int(version):04d}.csv"
    with open(ROOT / filename, "rb") as f:
        service.get_blob_client(store.container_name, name).upload_blob(f, overwrite=True)
    print(f"Copied {filename} to {DATASTORE}/{name}")


def show_row_counts(ml_client):
    for version in VERSIONS:
        asset = ml_client.data.get(DATA_ASSET, version=version)
        with tempfile.TemporaryDirectory() as tmp:
            # azureml-fsspec (the usual way to read azureml:// paths) has no Apple
            # Silicon build, so download through the workspace datastore instead.
            local_path = download_artifact_from_aml_uri(asset.path, tmp, ml_client.datastores)
            csv = next(Path(local_path).rglob("*.csv")) if Path(local_path).is_dir() else local_path
            df = pd.read_csv(csv)
        print(f"{DATA_ASSET} version {version}: {len(df)} rows ({asset.path})")


if __name__ == "__main__":
    client = get_ml_client()
    create_versions(client)
    show_row_counts(client)
