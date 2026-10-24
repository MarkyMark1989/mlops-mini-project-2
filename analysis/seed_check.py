"""Does the question 4 search pick the same settings when the CV folds change?

Runs the same randomized search as train.py --tune_iterations 30 (same search space,
30 candidates, 5-fold stratified CV repeated 3 times, F1 scoring) with the fold seed
cv_seed set to 0, 1, 2, 3 and 4, and prints the chosen settings, the CV F1 and the
test F1 for each. Nothing is logged to MLflow and no model file is written.

    .venv/bin/python src/data_prep.py --input_data telecom_churn_v2.csv
    .venv/bin/python analysis/seed_check.py      # about 10 minutes
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import TARGET, model_features  # noqa: E402
from train import SEARCH_SPACE  # noqa: E402

train = pd.read_csv(ROOT / "outputs/prepared/train.csv")
test = pd.read_csv(ROOT / "outputs/prepared/test.csv")
columns = model_features("improved")

chosen = {}
for cv_seed in range(5):
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        SEARCH_SPACE,
        n_iter=30,
        scoring="f1",
        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=cv_seed),
        random_state=42,
    )
    search.fit(train[columns], train[TARGET])
    test_f1 = f1_score(test[TARGET], search.best_estimator_.predict(test[columns]))
    params = dict(sorted(search.best_params_.items()))
    chosen[cv_seed] = str(params)
    print(f"cv_seed {cv_seed}: CV F1 {search.best_score_:.3f}, test F1 {test_f1:.3f}, {params}", flush=True)

counts = Counter(chosen.values())
settings, n = counts.most_common(1)[0]
print(f"\nSeeds 0 to 4: {n} of 5 chose {settings}")
