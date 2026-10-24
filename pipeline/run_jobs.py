"""Run the data preparation and training jobs on their own (Q2) as Azure ML command jobs.

Both jobs read the registered telecom-churn data asset rather than a local file and
log to the churn-final experiment. The training job does not register a model;
src/register.py does that (Q3, Q4).

    python pipeline/run_jobs.py                                    # Q2: data prep, then a default forest
    python pipeline/run_jobs.py --prep_job NAME --tune_iterations 30  # Q4: tuned forest on the same split
"""
import argparse
import sys
import time

from azure.ai.ml import Input, Output, command

from data_assets import DATA_ASSET
from workspace import ROOT, get_ml_client

sys.path.insert(0, str(ROOT / "src"))
from components import ENVIRONMENT  # noqa: E402
from features import FEATURE_SETS  # noqa: E402
from tracking import EXPERIMENT_NAME  # noqa: E402

SRC = str(ROOT / "src")
TERMINAL = {"Completed", "Failed", "Canceled", "NotResponding"}


def submit(ml_client, job):
    submitted = ml_client.jobs.create_or_update(job)
    print(f"Submitted {submitted.display_name} ({submitted.name}): {submitted.studio_url}", flush=True)
    # Poll instead of ml_client.jobs.stream, whose log download fails on this
    # workspace's storage with a base64 "Incorrect padding" error.
    while (status := ml_client.jobs.get(submitted.name).status) not in TERMINAL:
        time.sleep(20)
    print(f"{submitted.display_name}: {status}", flush=True)
    if status != "Completed":
        sys.exit(f"{submitted.display_name} did not complete; see {submitted.studio_url}")
    return submitted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", default="cpu-cluster")
    parser.add_argument("--data_version", default=None, help="data asset version; defaults to the latest")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--prep_job", default=None,
                        help="name of a completed data-prep job to train from, instead of running prep again")
    parser.add_argument("--feature_set", choices=sorted(FEATURE_SETS), default="improved")
    parser.add_argument("--tune_iterations", type=int, default=0,
                        help="randomized-search iterations; 0 trains a forest with default settings")
    parser.add_argument("--cv_seed", type=int, default=0)
    args = parser.parse_args()

    ml_client = get_ml_client()
    if args.prep_job:
        prep = ml_client.jobs.get(args.prep_job)
    else:
        prep = run_prep(ml_client, args)

    submit(ml_client, command(
        display_name="train-evaluate" if args.tune_iterations == 0 else "train-evaluate-tuned",
        experiment_name=EXPERIMENT_NAME,
        code=SRC,
        command="python train.py --data_dir ${{inputs.prepared_data}} --model_output ${{outputs.model_output}} "
                f"--feature_set {args.feature_set} --tune_iterations {args.tune_iterations} "
                f"--cv_seed {args.cv_seed}",
        # Command jobs only accept datastore paths as inputs; this is where the
        # prep job's prepared_data output lands by default.
        inputs={"prepared_data": Input(
            type="uri_folder",
            path=f"azureml://datastores/workspaceblobstore/paths/azureml/{prep.name}/prepared_data/")},
        outputs={"model_output": Output(type="uri_folder")},
        environment=ENVIRONMENT,
        compute=args.compute,
    ))


def run_prep(ml_client, args):
    version = args.data_version or ml_client.data.get(DATA_ASSET, label="latest").version
    print(f"Input data: {DATA_ASSET}:{version}")
    return submit(ml_client, command(
        display_name="data-prep",
        experiment_name=EXPERIMENT_NAME,
        code=SRC,
        command="python data_prep.py --input_data ${{inputs.input_data}} "
                "--output_dir ${{outputs.prepared_data}} --test_size ${{inputs.test_size}}",
        inputs={
            "input_data": Input(type="uri_file", path=f"azureml:{DATA_ASSET}:{version}"),
            "test_size": args.test_size,
        },
        outputs={"prepared_data": Output(type="uri_folder")},
        environment=ENVIRONMENT,
        compute=args.compute,
    ))


if __name__ == "__main__":
    main()
