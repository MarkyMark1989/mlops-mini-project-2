"""Exploratory analysis behind the preprocessing and feature engineering choices.

    .venv/bin/python analysis/eda.py      # prints findings, writes analysis/figures/*.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent
FIGURES = Path(__file__).resolve().parent / "figures"


def section(title):
    print(f"\n=== {title} ===")


def main():
    FIGURES.mkdir(exist_ok=True)
    v1 = pd.read_csv(ROOT / "telecom_churn_v1.csv")
    df = pd.read_csv(ROOT / "telecom_churn_v2.csv")
    pd.set_option("display.width", 120)

    section("Versions")
    print(f"v1: {len(v1)} rows, v2: {len(df)} rows")
    print("v1 is an exact prefix of v2:", df.head(len(v1)).equals(v1))

    section("Data quality")
    print("dtypes:", df.dtypes.value_counts().to_dict())
    print("missing values:", int(df.isna().sum().sum()), "| duplicate rows:", int(df.duplicated().sum()))
    print(df.describe().T.round(2))

    section("Target balance")
    print(df["Churn"].value_counts().rename({0: "stayed", 1: "churned"}).to_string())
    print(f"churn rate: {df['Churn'].mean():.1%}  -> stratify the split, judge by F1/precision/recall")

    section("Outliers (1.5 x IQR rule, continuous columns)")
    continuous = ["AccountWeeks", "DataUsage", "CustServCalls", "DayMins", "DayCalls",
                  "MonthlyCharge", "OverageFee", "RoamMins"]
    q1, q3 = df[continuous].quantile(0.25), df[continuous].quantile(0.75)
    iqr = q3 - q1
    outliers = ((df[continuous] < q1 - 1.5 * iqr) | (df[continuous] > q3 + 1.5 * iqr)).sum()
    print(outliers.to_string())
    print("Values are plausible (no negatives, no impossible values); kept, since tree models are robust to them.")

    section("Correlation with Churn")
    print(df.corr()["Churn"].drop("Churn").sort_values().round(3).to_string())

    section("Redundant features")
    corr = df.drop(columns="Churn").corr().abs()
    cols = corr.columns
    pairs = [(cols[i], cols[j], corr.iloc[i, j]) for i in range(len(cols))
             for j in range(i + 1, len(cols)) if corr.iloc[i, j] > 0.6]
    for a, b, r in pairs:
        print(f"{a} ~ {b}: |r| = {r:.3f}")
    X = df[["DayMins", "DataUsage", "OverageFee"]]
    fit = LinearRegression().fit(X, df["MonthlyCharge"])
    coefs = " + ".join(f"{c:.2f}*{n}" for n, c in zip(X.columns, fit.coef_))
    print(f"MonthlyCharge ~ {fit.intercept_:.2f} + {coefs}  (R^2 = {fit.score(X, df['MonthlyCharge']):.4f})")

    section("Churn rate by customer service calls")
    by_calls = df.groupby("CustServCalls")["Churn"].agg(rate="mean", customers="size")
    print(by_calls.round(3).to_string())
    high = df["CustServCalls"] >= 4
    print(f"0-3 calls: {df.loc[~high, 'Churn'].mean():.1%} churn | 4+ calls: {df.loc[high, 'Churn'].mean():.1%} churn")

    section("Churn rate by binary features")
    for col in ["ContractRenewal", "DataPlan"]:
        print(col, df.groupby(col)["Churn"].mean().round(3).to_dict())

    section("Average day call length")
    avg_call = df["DayMins"] / df["DayCalls"].replace(0, np.nan)
    print(f"AvgCallMins: corr with Churn = {avg_call.corr(df['Churn']):.3f}, "
          f"churned mean = {avg_call[df['Churn'] == 1].mean():.2f}, "
          f"stayed mean = {avg_call[df['Churn'] == 0].mean():.2f}")

    # Figures for the report
    fig, ax = plt.subplots(figsize=(8, 4))
    shown = by_calls[by_calls["customers"] >= 5]
    ax.bar(shown.index, shown["rate"], color=["#d62728" if c >= 4 else "#1f77b4" for c in shown.index])
    ax.axhline(df["Churn"].mean(), color="gray", linestyle="--", label="overall churn rate")
    ax.set(xlabel="Customer service calls", ylabel="Churn rate",
           title="Churn rate by customer service calls (4+ highlighted)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "churn_by_service_calls.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    full_corr = df.corr()
    im = ax.imshow(full_corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(full_corr)), full_corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(full_corr)), full_corr.columns)
    for i in range(len(full_corr)):
        for j in range(len(full_corr)):
            ax.text(j, i, f"{full_corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    fig.savefig(FIGURES / "correlation_matrix.png", dpi=150)

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, col in zip(axes.flat, continuous):
        ax.boxplot([df.loc[df["Churn"] == 0, col], df.loc[df["Churn"] == 1, col]])
        ax.set_xticks([1, 2], ["Stayed", "Churned"])
        ax.set_title(col)
    fig.suptitle("Feature distributions by churn outcome")
    fig.tight_layout()
    fig.savefig(FIGURES / "distributions_by_churn.png", dpi=150)

    print(f"\nFigures written to {FIGURES.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
