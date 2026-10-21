"""MLflow run handling that works both locally and inside Azure ML jobs."""
import os
from contextlib import contextmanager

import mlflow

EXPERIMENT_NAME = "telecom-churn"


def in_azureml_job():
    # Azure ML creates the MLflow run for a job and exposes its id here.
    return "MLFLOW_RUN_ID" in os.environ


@contextmanager
def start_run(run_name, job):
    """Start (or, in Azure ML, attach to) a run tagged with the job type."""
    if in_azureml_job():
        run = mlflow.start_run()
    else:
        mlflow.set_experiment(EXPERIMENT_NAME)
        run = mlflow.start_run(run_name=run_name)
    with run:
        mlflow.set_tag("job", job)
        yield run
