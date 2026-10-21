"""Training and evaluation job: random forest with MLflow autologging.

With --tune_iterations > 0 the forest is tuned with a randomized search
(5-fold stratified CV, optimizing F1) before evaluation on the test set.
"""
import argparse
import pickle
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import features
from features import MODEL_FEATURES, RAW_FEATURES, TARGET, add_features
from register import register_run_model
from tracking import start_run

SEARCH_SPACE = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 6, 10, 15, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", 0.5],
    "class_weight": [None, "balanced", "balanced_subsample"],
}


def parse_max_features(value):
    """CLI/pipeline values arrive as strings; '0.5' means a fraction, 'sqrt' a rule."""
    try:
        return float(value)
    except ValueError:
        return value


def build_serving_model(classifier):
    """Wrap the fitted forest so the saved model accepts raw predictor columns."""
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("classifier", classifier),
    ])


def log_confusion_matrix(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Stayed", "Churned"], cmap="Blues", ax=ax
    )
    ax.set_title("Test set confusion matrix")
    fig.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix_test.png")
    plt.close(fig)


def train_model(data_dir, model_output, n_estimators=100, max_depth=None,
                min_samples_leaf=1, max_features="sqrt", class_weight=None,
                tune_iterations=0, random_state=42, registered_model_name=None):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    X_train, y_train = train[MODEL_FEATURES], train[TARGET]
    X_test, y_test = test[MODEL_FEATURES], test[TARGET]

    # Test-set metrics are logged by hand below, and the deployable model is
    # logged separately because it must carry features.py with it.
    mlflow.sklearn.autolog(log_models=False, log_post_training_metrics=False)

    with start_run("train-evaluate" if tune_iterations == 0 else "train-evaluate-tuned", job="train") as run:
        if tune_iterations > 0:
            search = RandomizedSearchCV(
                RandomForestClassifier(random_state=random_state, n_jobs=-1),
                SEARCH_SPACE,
                n_iter=tune_iterations,
                scoring="f1",
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
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

        model = build_serving_model(classifier)
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
            )
            mlflow.log_artifacts(model_dir, artifact_path="model")
        # Set by the pipeline; standalone runs register via register.py instead.
        if registered_model_name:
            register_run_model(run.info.run_id, registered_model_name)

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
    args = parser.parse_args()
    train_model(**vars(args))
