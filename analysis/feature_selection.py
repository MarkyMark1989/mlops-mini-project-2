"""Which derived features actually improve the random forest?

Every comparison uses the same 5-fold stratified cross-validation, repeated five
times, on the training split only, and reports the gain fold by fold. The test split
is used once, at the end, to check the chosen feature set.

    .venv/bin/python src/data_prep.py --input_data telecom_churn_v2.csv   # creates outputs/prepared/
    .venv/bin/python analysis/feature_selection.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import RAW_FEATURES, TARGET  # noqa: E402

CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
MIN_GAIN = 0.005      # mean F1 gain needed to keep a feature
MIN_WINS = 18         # ...and it must win in at least 18 of the 25 folds

# Settings found by the randomized search in question 4.
TUNED = dict(n_estimators=100, max_depth=20, max_features="log2", min_samples_leaf=1,
             class_weight="balanced")


def add_candidates(df):
    df = df.copy()
    df["AvgCallMins"] = (df["DayMins"] / df["DayCalls"].replace(0, np.nan)).fillna(0.0)
    df["HighServiceCalls"] = (df["CustServCalls"] >= 4).astype(int)
    # The bill is about 0.48 + 0.17*DayMins + 10*DataUsage + 1.7*OverageFee (see eda.py).
    df["VoiceCharge"] = df["MonthlyCharge"] - 10 * df["DataUsage"]
    df["OverageShare"] = df["OverageFee"] / df["MonthlyCharge"]
    df["ServiceCallsPerYear"] = df["CustServCalls"] / (df["AccountWeeks"] / 52)
    df["NoRenewalCalls"] = (1 - df["ContractRenewal"]) * df["CustServCalls"]
    df["MinsPerDollar"] = df["DayMins"] / df["MonthlyCharge"]
    return df


CANDIDATES = ["AvgCallMins", "HighServiceCalls", "VoiceCharge", "OverageShare",
              "ServiceCallsPerYear", "NoRenewalCalls", "MinsPerDollar"]


def cv_f1(train, cols, params=None):
    model = RandomForestClassifier(random_state=42, n_jobs=-1, **(params or {}))
    return cross_val_score(model, train[cols], train[TARGET], cv=CV, scoring="f1")


def compare(new, base):
    diff = new - base
    return diff.mean(), int((diff > 0).sum()), diff.std()


def main():
    prepared = ROOT / "outputs" / "prepared"
    if not (prepared / "train.csv").exists():
        sys.exit("Run src/data_prep.py first to create outputs/prepared/.")
    train = add_candidates(pd.read_csv(prepared / "train.csv"))
    test = add_candidates(pd.read_csv(prepared / "test.csv"))

    base = cv_f1(train, RAW_FEATURES)
    print(f"Raw columns only: CV F1 {base.mean():.3f} ± {base.std():.3f}\n")

    print("Each candidate added on its own (gain over raw columns):")
    single = {}
    for c in CANDIDATES:
        scores = cv_f1(train, RAW_FEATURES + [c])
        gain, wins, sd = compare(scores, base)
        single[c] = gain
        print(f"  {c:20s} {gain:+.3f}  wins {wins:2d}/25  (sd {sd:.3f})")

    print("\nForward selection:")
    chosen, current = [], base
    remaining = sorted(CANDIDATES, key=lambda c: -single[c])
    while remaining:
        trials = {c: cv_f1(train, RAW_FEATURES + chosen + [c]) for c in remaining}
        best = max(trials, key=lambda c: trials[c].mean())
        gain, wins, _ = compare(trials[best], current)
        if gain < MIN_GAIN or wins < MIN_WINS:
            print(f"  stop: best next feature {best} adds {gain:+.3f} ({wins}/25 wins)")
            break
        chosen.append(best)
        remaining.remove(best)
        current = trials[best]
        print(f"  add {best:20s} {gain:+.3f}  wins {wins}/25  -> CV F1 {current.mean():.3f}")

    print("\nDropping one raw column from the chosen set (change in CV F1):")
    for col in RAW_FEATURES:
        cols = [c for c in RAW_FEATURES if c != col] + chosen
        gain, wins, _ = compare(cv_f1(train, cols), current)
        print(f"  without {col:16s} {gain:+.3f}  better in {wins:2d}/25")

    sets = {
        "Raw columns only": RAW_FEATURES,
        "Current (AvgCallMins + HighServiceCalls)": RAW_FEATURES + ["AvgCallMins", "HighServiceCalls"],
        f"Selected ({' + '.join(chosen) or 'none'})": RAW_FEATURES + chosen,
    }
    print("\nWith the tuned settings from question 4:")
    for name, cols in sets.items():
        s = cv_f1(train, cols, TUNED)
        print(f"  {name:55s} CV F1 {s.mean():.3f} ± {s.std():.3f}")

    print("\nTest set (used once), default and tuned settings:")
    for name, cols in sets.items():
        for label, params in (("default", {}), ("tuned", TUNED)):
            m = RandomForestClassifier(random_state=42, n_jobs=-1, **params).fit(train[cols], train[TARGET])
            p = m.predict(test[cols])
            print(f"  {name:55s} {label:7s} F1 {f1_score(test[TARGET], p):.3f}  "
                  f"precision {precision_score(test[TARGET], p):.3f}  recall {recall_score(test[TARGET], p):.3f}")


if __name__ == "__main__":
    main()
