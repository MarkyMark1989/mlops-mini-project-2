"""Build, submit and optionally schedule the end-to-end churn pipeline on Azure ML.

    python pipeline/run_pipeline.py --dry_run          # build and print the pipeline, no Azure calls
    python pipeline/run_pipeline.py --compute cpu-cluster [--create_compute] [--schedule]
"""
import argparse
import sys

from azure.ai.ml import Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import (AmlCompute, Environment, JobSchedule,
                                  RecurrencePattern, RecurrenceTrigger)
from azure.core.exceptions import ResourceNotFoundError

from data_assets import DATA_ASSET
from workspace import ROOT, get_ml_client

sys.path.insert(0, str(ROOT / "src"))
from components import (ENVIRONMENT_NAME, ENVIRONMENT_VERSION,  # noqa: E402
                        data_prep_component, train_component)
from register import MODEL_NAME  # noqa: E402
from tracking import EXPERIMENT_NAME  # noqa: E402

SCHEDULE_NAME = "telecom-churn-weekly-retrain"


@pipeline(
    name="telecom_churn_pipeline",
    description="Data prep -> tuned random forest training/evaluation -> model registration",
)
def churn_pipeline(input_data, test_size=0.2, tune_iterations=30, feature_set="improved"):
    prep = data_prep_component(input_data=input_data, test_size=test_size)
    train = train_component(
        prepared_data=prep.outputs.prepared_data,
        tune_iterations=tune_iterations,
        registered_model_name=MODEL_NAME,
        feature_set=feature_set,
    )
    return {"prepared_data": prep.outputs.prepared_data, "model": train.outputs.model_output}


def ensure_environment(ml_client):
    ml_client.environments.create_or_update(Environment(
        name=ENVIRONMENT_NAME,
        version=ENVIRONMENT_VERSION,
        description="Reusable environment for the telecom churn pipeline components",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
        conda_file=str(ROOT / "pipeline" / "conda.yaml"),
    ))


def ensure_compute(ml_client, name, create, size):
    try:
        ml_client.compute.get(name)
    except ResourceNotFoundError:
        if not create:
            sys.exit(f"Compute '{name}' not found. Pass --create_compute to create it.")
        # Scales to zero nodes when idle, so it only bills while jobs run.
        ml_client.compute.begin_create_or_update(AmlCompute(
            name=name, size=size, min_instances=0, max_instances=1,
            idle_time_before_scale_down=120,
        )).result()
        print(f"Created compute cluster {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", default="cpu-cluster")
    parser.add_argument("--create_compute", action="store_true")
    # 2 vCPUs is plenty for 3k rows. DSv3 family keeps it off the DSv2/FSv2 quota
    # used by the endpoint deployments.
    parser.add_argument("--compute_size", default="Standard_D2s_v3")
    parser.add_argument("--data_version", default="latest",
                        help="data asset version; 'latest' is resolved at submission time")
    parser.add_argument("--tune_iterations", type=int, default=30)
    parser.add_argument("--feature_set", choices=["baseline", "improved"], default="improved")
    parser.add_argument("--schedule", action="store_true",
                        help="create a weekly (Monday 06:00 UTC) retraining schedule instead of a one-off run")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    ml_client = None if args.dry_run else get_ml_client()
    data_version = args.data_version
    if data_version == "latest" and ml_client:
        # The job service rejects "name@latest" pipeline inputs, so resolve it here.
        # A schedule therefore keeps this version; rerun with --schedule after
        # registering new data to point the schedule at it.
        data_version = ml_client.data.get(DATA_ASSET, label="latest").version
    print(f"Input data: {DATA_ASSET}:{data_version}")

    job = churn_pipeline(
        input_data=Input(type="uri_file", path=f"azureml:{DATA_ASSET}:{data_version}"),
        tune_iterations=args.tune_iterations,
        feature_set=args.feature_set,
    )
    job.settings.default_compute = args.compute

    if args.dry_run:
        print(job)
        return

    ensure_environment(ml_client)
    ensure_compute(ml_client, args.compute, args.create_compute, args.compute_size)

    if not args.schedule:
        submitted = ml_client.jobs.create_or_update(job, experiment_name=EXPERIMENT_NAME)
        print(f"Submitted pipeline job {submitted.name}: {submitted.studio_url}")
    else:
        # Schedule only: an immediate run too would register an extra model version.
        schedule = JobSchedule(
            name=SCHEDULE_NAME,
            trigger=RecurrenceTrigger(
                frequency="week", interval=1,
                schedule=RecurrencePattern(week_days=["monday"], hours=6, minutes=0),
                time_zone="UTC",
            ),
            create_job=job,
        )
        ml_client.schedules.begin_create_or_update(schedule).result()
        print(f"Created schedule {SCHEDULE_NAME}; disable it with "
              f"ml_client.schedules.begin_disable('{SCHEDULE_NAME}') when no longer needed")


if __name__ == "__main__":
    main()
