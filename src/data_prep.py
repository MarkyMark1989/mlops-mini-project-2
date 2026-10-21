"""Data preparation job: clean, impute, engineer features and split train/test."""
import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from features import DERIVED_FEATURES, RAW_FEATURES, TARGET, add_features, clean
from tracking import start_run


def read_csv(path):
    """Read a CSV given either a file path or a folder (Azure ML uri_folder input)."""
    path = Path(path)
    if path.is_dir():
        path = sorted(path.glob("*.csv"))[0]
    return pd.read_csv(path)


def prepare_data(input_data, output_dir, test_size=0.2, random_state=42):
    with start_run("data-prep", job="data_prep"):
        raw = read_csv(input_data)
        df, steps = clean(raw)

        # Stratify so both splits keep the ~14% churn rate.
        train, test = train_test_split(
            df, test_size=test_size, stratify=df[TARGET], random_state=random_state
        )
        steps.append(f"stratified_train_test_split: test_size={test_size}")

        # Medians come from the training split only, so the test set never leaks into them.
        medians = train[RAW_FEATURES].median()
        n_missing = int(train[RAW_FEATURES].isna().sum().sum() + test[RAW_FEATURES].isna().sum().sum())
        train = train.fillna(medians)
        test = test.fillna(medians)
        steps.append(f"median_imputation: filled {n_missing} missing feature values using train medians")

        train = pd.concat([add_features(train), train[TARGET]], axis=1)
        test = pd.concat([add_features(test), test[TARGET]], axis=1)
        steps.append(f"feature_engineering: added {', '.join(DERIVED_FEATURES)}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train.to_csv(output_dir / "train.csv", index=False)
        test.to_csv(output_dir / "test.csv", index=False)

        mlflow.log_params({
            "preprocessing": "schema_validation,range_checks,drop_duplicates,"
                             "drop_missing_target,stratified_split,median_imputation",
            "derived_features": ",".join(DERIVED_FEATURES),
            "test_size": test_size,
            "random_state": random_state,
        })
        mlflow.log_metrics({
            "input_rows": len(raw),
            "clean_rows": len(df),
            "train_rows": len(train),
            "test_rows": len(test),
            "train_churn_rate": train[TARGET].mean(),
            "test_churn_rate": test[TARGET].mean(),
        })
        mlflow.log_dict({"steps": steps, "train_medians": medians.to_dict()}, "preprocessing.json")

        print(json.dumps(steps, indent=2))
        print(f"train={len(train)} rows, test={len(test)} rows -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", required=True)
    parser.add_argument("--output_dir", default="outputs/prepared")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    prepare_data(args.input_data, args.output_dir, args.test_size, args.random_state)
