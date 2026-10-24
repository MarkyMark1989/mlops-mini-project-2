"""Training and evaluation job: random forest with MLflow autologging.

With --tune_iterations > 0 the forest is tuned with a randomized search
(5-fold stratified CV repeated 3 times, optimizing F1) before evaluation on the
test set. --feature_set picks the derived features (see features.FEATURE_SETS);
"improved" (raw columns + VoiceCharge) is the default, "baseline" the original set.
--cv_seed fixes the CV splits; with the default of 0 the improved search picks
the same settings under 4 of 5 split seeds (seed 42 was the odd one out).
"""
import argparse
import hashlib
import json
import pickle
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import features
from features import FEATURE_SETS, RAW_FEATURES, TARGET, add_features, model_features
from register import production_version, register_run_model
from tracking import start_run

SEARCH_SPACE = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 6, 10, 15, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", 0.5],
    "class_weight": [None, "balanced", "balanced_subsample"],
}
# Saved inside every model folder: hashes of the rows the model was trained on.
FINGERPRINT_FILE = "training_rows.json"
# With fewer unseen test rows than this, a fair comparison is too noisy to use.
MIN_FAIR_ROWS = 100
# Azure's no-code MLflow deployment imports this package but doesn't install it.
EXTRA_PIP_REQUIREMENTS = ["azureml-ai-monitoring==1.0.0"]


def parse_max_features(value):
    """CLI/pipeline values arrive as strings; '0.5' means a fraction, 'sqrt' a rule."""
    try:
        return float(value)
    except ValueError:
        return value


def build_serving_model(classifier, columns):
    """Wrap the fitted forest so the saved model accepts raw predictor columns."""
    return Pipeline([
        ("features", FunctionTransformer(add_features, kw_args={"columns": columns})),
        ("classifier", classifier),
    ])


def row_fingerprints(df):
    """A short SHA-256 hash of each row's raw predictors and target.

    Values are formatted the same way for every data version, so a customer who
    appears in data v1 and v2 gets the same hash in both.
    """
    values = df[RAW_FEATURES + [TARGET]].astype(float).to_numpy()
    return [hashlib.sha256(",".join(f"{v:.10g}" for v in row).encode()).hexdigest()[:16]
            for row in values]


def log_confusion_matrix(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Stayed", "Churned"], cmap="Blues", ax=ax
    )
    ax.set_title("Test set confusion matrix")
    fig.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix_test.png")
    plt.close(fig)


def beats_production(model_name, candidate, X_raw, y, test_fingerprints):
    """Compare F1 with the production model on test rows it never trained on.

    A new data version still contains the customers the production model was
    trained on, and scoring it on those rows would inflate its F1. Each saved
    model carries the fingerprints of its training rows, so both models are
    scored only on the test rows missing from them. If the production model has
    no fingerprints, or fewer than MIN_FAIR_ROWS test rows (or no churners) are
    unseen, both are scored on the full test set instead. That favors the
    production model, so a candidate that still wins is genuinely better.
    A tie keeps the production model.
    """
    current = production_version(model_name)
    if current is None:
        print(f"No production version of {model_name} yet")
        return True
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(mlflow.artifacts.download_artifacts(
            f"models:/{model_name}/{current.version}", dst_path=tmp))
        production = mlflow.pyfunc.load_model(str(model_dir))
        fingerprint_file = model_dir / FINGERPRINT_FILE
        seen = set(json.loads(fingerprint_file.read_text())["fingerprints"]) if fingerprint_file.exists() else None

    y = y.to_numpy()
    unseen = np.array([fp not in seen for fp in test_fingerprints]) if seen is not None else None
    if unseen is None:
        mask, check = np.ones(len(y), bool), "fallback: production model has no training fingerprints"
    elif unseen.sum() < MIN_FAIR_ROWS or y[unseen].sum() == 0:
        mask, check = np.ones(len(y), bool), f"fallback: only {unseen.sum()} unseen test rows"
    else:
        mask, check = unseen, "unseen test rows only"

    candidate_f1 = f1_score(y[mask], candidate.predict(X_raw[mask]))
    production_f1 = f1_score(y[mask], production.predict(X_raw[mask]))
    mlflow.log_metrics({
        "fair_eval_rows": int(mask.sum()),
        "fair_eval_excluded": int((~mask).sum()),
        "candidate_fair_f1": candidate_f1,
        "production_fair_f1": production_f1,
    })
    mlflow.set_tags({"production_version": current.version, "fair_check": check})
    print(f"Comparison on {mask.sum()} of {len(y)} test rows ({check}): candidate F1 {candidate_f1:.4f}, "
          f"production (version {current.version}) {production_f1:.4f}")
    return candidate_f1 > production_f1


def train_model(data_dir, model_output, n_estimators=100, max_depth=None,
                min_samples_leaf=1, max_features="sqrt", class_weight=None,
                tune_iterations=0, random_state=42, registered_model_name=None,
                feature_set="improved", cv_seed=0):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    columns = model_features(feature_set)
    X_train, y_train = train[columns], train[TARGET]
    X_test, y_test = test[columns], test[TARGET]

    # Test-set metrics are logged by hand below, and the deployable model is
    # logged separately because it must carry features.py with it.
    mlflow.sklearn.autolog(log_models=False, log_post_training_metrics=False)

    with start_run("train-evaluate" if tune_iterations == 0 else "train-evaluate-tuned", job="train") as run:
        mlflow.log_params({"feature_set": feature_set,
                           "derived_features": ",".join(FEATURE_SETS[feature_set])})
        if tune_iterations > 0:
            search = RandomizedSearchCV(
                RandomForestClassifier(random_state=random_state, n_jobs=-1),
                SEARCH_SPACE,
                n_iter=tune_iterations,
                scoring="f1",
                cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=cv_seed),
                random_state=random_state,
            )
            search.fit(X_train, y_train)
            classifier = search.best_estimator_
            print("Best CV F1:", round(search.best_score_, 4), "params:", search.best_params_)
        else:
            classifier = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                min_samples_leaf=min_samples_leaf, max_features=max_features,
                class_weight=class_weight, random_state=random_state, n_jobs=-1,
            )
            classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)[:, 1]
        test_metrics = {
            "test_f1": f1_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(test_metrics)
        log_confusion_matrix(y_test, y_pred)

        model = build_serving_model(classifier, columns)
        # Declare inputs as floats so API callers may send 128 or 128.0.
        raw_example = test[RAW_FEATURES].astype(float)
        # Azure ML's MLflow server lacks MLflow 3's logged-model API (log_model
        # gets a 404), so save the model locally and upload it as run artifacts.
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = str(Path(tmp) / "model")  # azureml-mlflow needs str, not Path
            mlflow.sklearn.save_model(
                model,
                model_dir,
                code_paths=[features.__file__],
                # Both types come from this project's own training code.
                skops_trusted_types=["features.add_features", "sklearn.tree._tree.Tree"],
                signature=infer_signature(raw_example, model.predict(raw_example)),
                input_example=raw_example.head(3),
                extra_pip_requirements=EXTRA_PIP_REQUIREMENTS,
            )
            fingerprints = sorted(set(row_fingerprints(train)))
            (Path(model_dir) / FINGERPRINT_FILE).write_text(json.dumps({
                "columns": RAW_FEATURES + [TARGET],
                "hash": "sha256 of the comma-joined values, first 16 hex characters",
                "rows": len(train),
                "fingerprints": fingerprints,
            }))
            mlflow.log_artifacts(model_dir, artifact_path="model")
        # Set by the pipeline, which only registers a model that beats production.
        # Standalone runs register via register.py instead.
        if registered_model_name:
            if beats_production(registered_model_name, model, raw_example, y_test, row_fingerprints(test)):
                version = register_run_model(run.info.run_id, registered_model_name)
                decision = f"registered as version {version.version}"
            else:
                decision = "skipped: not better than production"
            mlflow.set_tag("registration", decision)
            print("Registration:", decision)

        model_output = Path(model_output)
        model_output.mkdir(parents=True, exist_ok=True)
        with open(model_output / "model.pkl", "wb") as f:
            pickle.dump(model, f)

        for name, value in test_metrics.items():
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="outputs/prepared")
    parser.add_argument("--model_output", default="outputs/model")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--max_features", type=parse_max_features, default="sqrt")
    parser.add_argument("--class_weight", default=None)
    parser.add_argument("--tune_iterations", type=int, default=0)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--registered_model_name", default=None)
    parser.add_argument("--feature_set", choices=sorted(FEATURE_SETS), default="improved")
    parser.add_argument("--cv_seed", type=int, default=0)
    args = parser.parse_args()
    train_model(**vars(args))
