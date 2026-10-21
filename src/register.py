"""Find the latest training run by experiment name/id and register its model."""
import argparse

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from tracking import EXPERIMENT_NAME

MODEL_NAME = "telecom-churn-rf"


def register_run_model(run_id, model_name=MODEL_NAME):
    """Register the 'model' artifact of a run as a new version of model_name.

    mlflow.register_model calls MLflow 3's logged-model API, which Azure ML's
    MLflow server doesn't implement, so use the registry calls directly.
    """
    client = MlflowClient()
    try:
        client.get_registered_model(model_name)
    except MlflowException:
        client.create_registered_model(model_name)
    # Azure rejects runs:/ sources and wants the run's own artifact location.
    source = f"{client.get_run(run_id).info.artifact_uri}/model"
    return client.create_model_version(model_name, source=source, run_id=run_id)


def register_latest_model(experiment_name=EXPERIMENT_NAME, model_name=MODEL_NAME, dry_run=False):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No experiment named {experiment_name!r}")
    print(f"Experiment {experiment.name!r} has id {experiment.experiment_id}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.job = 'train'",
        order_by=["attributes.start_time DESC"],
        max_results=20,
    )
    # Azure ML's MLflow server ignores status filters (returns nothing), so skip
    # failed or still-running runs here instead.
    finished = [r for r in runs if r.info.status == "FINISHED"]
    if not finished:
        raise ValueError("No finished training runs found")
    # Azure's search results omit metrics; fetch the full run.
    run = client.get_run(finished[0].info.run_id)
    print(f"Latest training run: {run.info.run_name} ({run.info.run_id}), "
          f"test_f1={run.data.metrics.get('test_f1'):.4f}")

    if dry_run:
        print("Dry run: not registering")
        return None
    version = register_run_model(run.info.run_id, model_name)
    print(f"Registered {model_name} version {version.version}")
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default=EXPERIMENT_NAME)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--dry_run", action="store_true", help="find the run but don't register it")
    args = parser.parse_args()
    register_latest_model(args.experiment_name, args.model_name, args.dry_run)
