"""Nested-CV check: does the hyperparameter search (model v2) beat the default forest (model v1)?

Both use the raw columns plus VoiceCharge. The original derived pair (AvgCallMins,
HighServiceCalls) and the raw columns alone are included for comparison.

The search runs inside each outer training fold and is scored on that fold's held-out
part, so the tuned result is not flattered. The test split is used once at the end.

    .venv/bin/python src/data_prep.py --input_data telecom_churn_v2.csv
    .venv/bin/python analysis/tuning_check.py      # about 10-15 minutes
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import RAW_FEATURES, TARGET  # noqa: E402
from train import SEARCH_SPACE  # noqa: E402

train = pd.read_csv(ROOT / "outputs/prepared/train.csv")
test = pd.read_csv(ROOT / "outputs/prepared/test.csv")
ORIGINAL = RAW_FEATURES + ["AvgCallMins", "HighServiceCalls"]
NEW = RAW_FEATURES + ["VoiceCharge"]

outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
rows = []
X, y = train, train[TARGET]
for k, (tr, va) in enumerate(outer.split(X, y)):
    Xtr, Xva, ytr, yva = X.iloc[tr], X.iloc[va], y.iloc[tr], y.iloc[va]
    f1 = lambda m, cols: f1_score(yva, m.fit(Xtr[cols], ytr).predict(Xva[cols]))
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1), SEARCH_SPACE, n_iter=30, scoring="f1",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=k), random_state=k)
    search.fit(Xtr[NEW], ytr)
    rows.append({
        "v1: VoiceCharge, default settings": f1(RandomForestClassifier(random_state=42, n_jobs=-1), NEW),
        "v2: VoiceCharge + search (nested)": f1_score(yva, search.predict(Xva[NEW])),
        "Raw columns only, default": f1(RandomForestClassifier(random_state=42, n_jobs=-1), RAW_FEATURES),
        "Original pair, default": f1(RandomForestClassifier(random_state=42, n_jobs=-1), ORIGINAL),
        "chosen": search.best_params_,
    })
    print(f"outer fold {k + 1}/10 done", flush=True)

res = pd.DataFrame(rows)
base = res.iloc[:, 0]
print("\nOuter-fold F1 (10 folds):")
for col in res.columns[:-1]:
    d = res[col] - base
    print(f"  {col:45s} {res[col].mean():.3f} ± {res[col].std():.3f}   vs v1 {d.mean():+.3f}, better in {(d > 0).sum()}/10")
print("\nSettings chosen in each outer fold:")
for p in res["chosen"]:
    print("  ", p)

# Final model: search on the whole training split, then one look at the test set.
final = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1), SEARCH_SPACE, n_iter=30, scoring="f1",
    cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0), random_state=42)
final.fit(train[NEW], train[TARGET])
print(f"\nFinal search on full training split: best CV F1 {final.best_score_:.3f}, params {final.best_params_}")
for name, model, cols in [
    ("v1: VoiceCharge, default", RandomForestClassifier(random_state=42, n_jobs=-1), NEW),
    ("v2: VoiceCharge + search", final.best_estimator_, NEW),
    ("Original pair, default", RandomForestClassifier(random_state=42, n_jobs=-1), ORIGINAL),
]:
    p = model.predict(test[cols]) if model is final.best_estimator_ else model.fit(train[cols], train[TARGET]).predict(test[cols])
    print(f"  test  {name:28s} F1 {f1_score(test[TARGET], p):.3f}  precision {precision_score(test[TARGET], p):.3f}  recall {recall_score(test[TARGET], p):.3f}")
