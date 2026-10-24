# Telecom customer churn prediction on Azure ML

Code for MLOps Mini Project 2. The project predicts whether a telecom customer will cancel their service and covers data versioning, MLflow tracking, the model registry, an Azure ML pipeline built with mldesigner, a Flask API, an Azure ML online endpoint and a Streamlit app.

## Contents

| Path | What it is | Question |
|---|---|---|
| `telecom_churn_v1.csv`, `telecom_churn_v2.csv`, `DataDictionary-1.xlsx` | Data and column descriptions | |
| `analysis/eda.py` | Exploratory analysis; writes the charts in `analysis/figures/` | 2 |
| `analysis/feature_selection.py` | Compares candidate derived features with repeated cross-validation | 2, 4 |
| `analysis/tuning_check.py` | Nested cross-validation of version 2 (tuned) against version 1 (default) | 4 |
| `analysis/seed_check.py` | Reruns the question 4 search with five fold seeds and prints the settings each one picks | 4 |
| `src/features.py` | Column definitions, cleaning, the derived features and the `improved` (default) and `baseline` feature sets | 2, 6, 7 |
| `src/data_prep.py` | Data preparation job | 2 |
| `src/train.py` | Training and evaluation job, optional randomized search, training-row fingerprints and the pipeline's production check | 2, 4, 5 |
| `src/register.py` | Finds the latest training run and registers its model; `--promote` marks the production version | 3, 4 |
| `src/tracking.py` | MLflow run handling for local runs and Azure ML jobs | 2 |
| `src/components.py` | Pipeline components (mldesigner `@command_component`) | 5 |
| `pipeline/data_assets.py` | Creates data asset versions 1 and 2, counts their rows and copies them to the pipeline's landing folder | 1, 5 |
| `pipeline/run_jobs.py` | Runs the data preparation and (optionally tuned) training jobs in Azure ML on the data asset | 2, 4 |
| `pipeline/run_pipeline.py`, `pipeline/conda.yaml` | Builds, runs and schedules the pipeline in a reusable environment | 5 |
| `pipeline/workspace.py` | Workspace connection; prints the MLflow tracking URI | 1 to 5 |
| `api/app.py` | Flask API with Swagger UI | 6 |
| `api/export_model.py` | Saves a registered version as the pickle the Flask API serves | 6 |
| `deployment/deploy.py`, `score.py`, `conda.yaml` | Online endpoint deployment, scoring script and inference environment | 7 |
| `deployment/count_split.py` | Counts which deployment answers through the traffic split | 7e |
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

Run these from the project root. MLflow stores local runs in `./mlflow.db`.

```bash
.venv/bin/python analysis/eda.py
.venv/bin/python src/data_prep.py --input_data telecom_churn_v2.csv
.venv/bin/python src/train.py                         # default random forest on raw columns + VoiceCharge
.venv/bin/python src/train.py --tune_iterations 30    # randomized search, 5-fold CV repeated 3 times
.venv/bin/python analysis/feature_selection.py        # feature comparison
.venv/bin/python analysis/tuning_check.py             # nested CV, version 2 against version 1
.venv/bin/python analysis/seed_check.py               # search under fold seeds 0 to 4
.venv/bin/python api/app.py                           # Swagger UI at http://127.0.0.1:5001/apidocs/
```

The Flask API serves `outputs/model/model.pkl`. `train.py` writes that file on every run, so after the commands above it holds the tuned model, which has version 2's settings. For the report we served registered version 2 itself, exported with `api/export_model.py` (see below). Set `MODEL_PATH` to serve a different pickle.

## Running on Azure ML

```bash
.venv/bin/python pipeline/data_assets.py                                  # Q1: data asset v1/v2, row counts, landing folder
.venv/bin/python pipeline/run_jobs.py                                     # Q2: data prep + default forest in experiment churn-final
export MLFLOW_TRACKING_URI=$(.venv/bin/python pipeline/workspace.py)       # point local scripts at the workspace
.venv/bin/python src/register.py                                          # Q3: registers the latest run as churn-rf
.venv/bin/python pipeline/run_jobs.py --prep_job <prep job> --tune_iterations 30   # Q4: tuned forest on the same split
.venv/bin/python src/register.py                                          # registers it as the next churn-rf version
.venv/bin/python src/register.py --promote 2                              # marks version 2 as production
.venv/bin/python api/export_model.py --version 2                          # Q6: version 2 as outputs/model/model.pkl for Flask
.venv/bin/python pipeline/run_pipeline.py --compute cpu-cluster           # Q5: one pipeline run
.venv/bin/python pipeline/run_pipeline.py --compute cpu-cluster --schedule  # weekly retraining schedule
```

The compute cluster scales down to zero nodes when it is idle. Pass `--create_compute` to `run_pipeline.py` if the cluster does not exist yet.

The pipeline reads the newest file in the datastore folder `telecom-churn/landing/`, so scheduled runs pick up new data versions. Its training step registers a model only if it beats the production version, scored only on test rows the production model never trained on (each saved model carries a fingerprint of its training rows).

Deployment and the Streamlit app (endpoint names must be unique within the Azure region):

```bash
.venv/bin/python deployment/deploy.py create    --endpoint <name> --version 1
.venv/bin/python deployment/deploy.py add-green --endpoint <name> --version 2 --traffic 30
.venv/bin/python deployment/deploy.py test      --endpoint <name> --deployment green
.venv/bin/python deployment/deploy.py secrets   --endpoint <name>   # saves the URI and key for Streamlit
.venv/bin/python deployment/count_split.py                          # 100 requests; which deployment answered
.venv/bin/streamlit run frontend/streamlit_app.py
.venv/bin/python deployment/deploy.py delete    --endpoint <name>
```

Online endpoints are billed for as long as they exist, so delete the endpoint when you are done with it.

## Results

All runs are in the MLflow experiment `churn-final`, and both models are versions of the registered model `churn-rf`. Both use the ten raw columns plus VoiceCharge (MonthlyCharge minus 10 x DataUsage).

| Version | Model | Test F1 / precision / recall |
|---|---|---|
| 1 | Default random forest | 0.705 / 0.785 / 0.639 |
| 2 | Settings from the randomized search (production) | 0.713 / 0.805 / 0.639 |

On the 667-row test set each catches 62 of 97 churners, 61 of them the same, and version 2 raises two fewer false alarms. In nested cross-validation version 2 scores 0.776 against 0.769 for version 1. Most of the gain over the original features comes from VoiceCharge, which adds 0.023 to cross-validated F1 over the raw columns; on the small test set that gain does not show.
