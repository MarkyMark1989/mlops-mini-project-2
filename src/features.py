"""Schema, cleaning and feature engineering shared by every stage.

The data prep job, the saved model pipeline, the Flask API and the Azure ML
deployment all import this module, so derived features are computed the same
way in training and at inference time.
"""
import numpy as np
import pandas as pd

TARGET = "Churn"
BINARY_FEATURES = ["ContractRenewal", "DataPlan"]
RAW_FEATURES = [
    "AccountWeeks", "ContractRenewal", "DataPlan", "DataUsage", "CustServCalls",
    "DayMins", "DayCalls", "MonthlyCharge", "OverageFee", "RoamMins",
]
DERIVED_FEATURES = ["AvgCallMins", "HighServiceCalls", "VoiceCharge"]
# Derived features each model uses on top of the raw columns. "improved" is the
# default; it came out of analysis/feature_selection.py, where only VoiceCharge
# helped. "baseline" is the original pair, which did not improve on raw columns.
FEATURE_SETS = {
    "baseline": ["AvgCallMins", "HighServiceCalls"],
    "improved": ["VoiceCharge"],
}

# Churn rate is ~11% for 0-3 customer service calls and ~52% for 4 or more.
HIGH_SERVICE_CALLS_THRESHOLD = 4


def clean(df):
    """Validate and clean raw data. Returns (cleaned_df, list of step descriptions)."""
    missing = set(RAW_FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Input data is missing columns: {sorted(missing)}")

    steps = []
    df = df[RAW_FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")
    steps.append("schema_validation: kept the 11 expected columns, coerced to numeric")

    for col in BINARY_FEATURES + [TARGET]:
        df.loc[~df[col].isin([0, 1]), col] = np.nan
    numeric = [c for c in RAW_FEATURES if c not in BINARY_FEATURES]
    df[numeric] = df[numeric].mask(df[numeric] < 0)
    steps.append("range_checks: non-0/1 binary values and negative usage values set to missing")

    before = len(df)
    df = df.drop_duplicates()
    steps.append(f"drop_duplicates: removed {before - len(df)} rows")

    before = len(df)
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    steps.append(f"drop_missing_target: removed {before - len(df)} rows")

    return df.reset_index(drop=True), steps


def model_features(feature_set):
    """Columns a model trained on the given feature set expects, in order."""
    return RAW_FEATURES + FEATURE_SETS[feature_set]


def add_features(df, columns=None):
    """Return model input columns (raw + derived features) for raw input rows.

    `columns` picks the columns to return. The serving model passes its own list,
    because a forest fitted with feature names rejects extra columns. Without it
    the baseline columns are returned, which is what models saved before feature
    sets existed expect.
    """
    out = df[RAW_FEATURES].astype(float)
    # Average length of a daytime call; customers with no day calls get 0.
    out["AvgCallMins"] = (out["DayMins"] / out["DayCalls"].replace(0, np.nan)).fillna(0.0)
    out["HighServiceCalls"] = (out["CustServCalls"] >= HIGH_SERVICE_CALLS_THRESHOLD).astype(int)
    # The bill is about 0.48 + 0.17*DayMins + 10*DataUsage + 1.7*OverageFee, so
    # removing the $10/GB data part leaves the voice part of the monthly charge.
    out["VoiceCharge"] = out["MonthlyCharge"] - 10 * out["DataUsage"]
    return out[list(columns or model_features("baseline"))]
