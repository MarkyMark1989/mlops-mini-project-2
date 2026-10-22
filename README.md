# Telecom customer churn prediction on Azure ML

Code for MLOps Mini Project 2. The project predicts whether a telecom customer will cancel their service and covers data versioning, MLflow tracking, the model registry, an Azure ML pipeline built with mldesigner, a Flask API, an Azure ML online endpoint and a Streamlit app.

## Contents

| Path | What it is | Question |
|---|---|---|
| `telecom_churn_v1.csv`, `telecom_churn_v2.csv`, `DataDictionary-1.xlsx` | Data and column descriptions | |
| `analysis/eda.py` | Exploratory analysis; writes the charts in `analysis/figures/` | 2a |
| `analysis/feature_selection.py` | Compares candidate derived features with repeated cross-validation | 2a, 4 |
| `analysis/tuning_check.py` | Nested cross-validation check of the improved model and its search | 4 |
| `src/features.py` | Column definitions, cleaning, the derived features and the `baseline`/`improved` feature sets | 2a, 4 |
| `src/data_prep.py` | Data preparation job | 2a |
| `src/train.py` | Training and evaluation job, with a `--feature_set` option and optional randomized search | 2b, 4 |
| `src/register.py` | Finds the latest training run and registers its model | 3 |
| `src/tracking.py` | MLflow run handling for local runs and Azure ML jobs | 2 |
| `src/components.py` | Pipeline components (mldesigner `@command_component`) | 5 |
| `pipeline/data_assets.py` | Creates data asset versions 1 and 2 and counts their rows | 1 |
| `pipeline/run_jobs.py` | Runs the data preparation and training jobs in Azure ML on the data asset | 2 |
| `pipeline/run_pipeline.py`, `pipeline/conda.yaml` | Builds, runs and schedules the pipeline in a reusable environment | 5 |
| `pipeline/workspace.py` | Workspace connection; prints the MLflow tracking URI | 1 to 5 |
| `api/app.py` | Flask API with Swagger UI | 6 |
| `deployment/deploy.py`, `score.py`, `conda.yaml` | Online endpoint deployment, scoring script and inference environment | 7 |
| `frontend/streamlit_app.py` | Streamlit app that calls the Azure endpoint | 8 |
| `report/make_diagram.py` | Draws the process flow diagram for 7d | 7d |

## Setup

Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The Azure steps also need the Azure CLI with the `ml` extension (`az login`), and the workspace's `config.json` from Azure ML Studio in the project root.

## Running locally

Run these from the project root. MLflow stores runs in `./mlflow.db`.

```bash
.venv/bin/python analysis/eda.py
.venv/bin/python src/data_prep.py --input_data telecom_churn_v2.csv
.venv/bin/python src/train.py                         # baseline model
.venv/bin/python src/register.py                      # registers it as telecom-churn-rf version 1
.venv/bin/python src/train.py --tune_iterations 30    # tuned model
.venv/bin/python src/register.py                      # registers it as version 2
.venv/bin/python src/train.py --feature_set improved --tune_iterations 30   # improved model (VoiceCharge)
.venv/bin/python src/register.py                      # registered on Azure as version 5
.venv/bin/python api/app.py                           # Swagger UI at http://127.0.0.1:5001/apidocs/
```

The Flask API serves `outputs/model/model.pkl`, which is the model from whichever `train.py` run finished last. In the order above that is the improved model. To serve the baseline instead (as in the report), run `src/train.py` with no options again before starting the API.

Version 2 was tuned with plain 5-fold CV. The search now uses 5-fold CV repeated three times, so the second `train.py` command no longer reproduces version 2 exactly.

## Running on Azure ML

```bash
.venv/bin/python pipeline/data_assets.py                                  # Q1
.venv/bin/python pipeline/run_jobs.py --compute cpu-cluster               # Q2 as Azure ML jobs on the data asset
export MLFLOW_TRACKING_URI=$(.venv/bin/python pipeline/workspace.py)       # log local runs (register.py, tuning) to the workspace
.venv/bin/python pipeline/run_pipeline.py --compute cpu-cluster --create_compute   # --feature_set defaults to improved
.venv/bin/python pipeline/run_pipeline.py --compute cpu-cluster --schedule  # weekly retraining schedule
```

The compute cluster scales down to zero nodes when it is idle.

Deployment and the Streamlit app (endpoint names must be unique within the Azure region):

```bash
.venv/bin/python deployment/deploy.py create    --endpoint <name> --version 1
.venv/bin/python deployment/deploy.py add-green --endpoint <name> --version 5 --traffic 30
.venv/bin/python deployment/deploy.py test      --endpoint <name>
.venv/bin/python deployment/deploy.py secrets   --endpoint <name>   # saves the URI and key for Streamlit
.venv/bin/streamlit run frontend/streamlit_app.py
.venv/bin/python deployment/deploy.py delete    --endpoint <name>
```

Online endpoints are billed for as long as they exist, so delete the endpoint when you are done with it.

## Results

On the 667-row test set, the baseline random forest (model version 1) scored F1 0.698, precision 0.800 and recall 0.619. The first tuned model (version 2) scored F1 0.660, precision 0.650 and recall 0.670. Nested cross-validation showed that its tuning did not help (0.744 against 0.747 for version 1).

The improved model (version 5) replaces the two original derived features with VoiceCharge (MonthlyCharge minus 10 × DataUsage) and was tuned with repeated cross-validation. It scored F1 0.713, precision 0.805 and recall 0.639 on the test set, and 0.776 in nested cross-validation, beating version 1 in 7 of 10 outer folds. The pipeline rebuilds the same model as version 6.
