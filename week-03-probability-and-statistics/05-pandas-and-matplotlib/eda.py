"""Week 3.5 — a full EDA workflow with pandas and Matplotlib on a messy churn dataset."""
from pathlib import Path

import numpy as np
import pandas as pd

from churn_data import make_churn

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def section(title):
    print(f"\n=== {title} ===")


# 1. Load (we write the raw CSV first so we practise read_csv like on a real project)
raw_path = OUT / "churn_raw.csv"
make_churn().to_csv(raw_path, index=False)
df = pd.read_csv(raw_path, parse_dates=["signup_date"])

section("2. Shape and dtypes")
print(df.shape)
print(df.dtypes)
print("-> total_charges is text (object/str dtype), not float — something non-numeric is hiding in it")

section("3. Peek")
print(df.sample(5, random_state=1))

section("4. Missing values (fraction per column)")
print(df.isna().mean().round(3)[lambda s: s > 0])

section("5. Duplicates")
print("fully duplicated rows:", df.duplicated().sum())
print("duplicated customer_id:", df.customer_id.duplicated().sum())
df = df.drop_duplicates("customer_id").reset_index(drop=True)

section("6. Fix types: total_charges text -> number")
print("NaNs already in total_charges:", df.total_charges.isna().sum(),
      "<- read_csv turns 'N/A', 'NA', '' etc. into NaN by default (na_values)")
bad = df.total_charges.dropna()
bad = bad[pd.to_numeric(bad, errors="coerce").isna()]
print("other non-numeric values:", bad.value_counts().to_dict(), "<- e.g. a lone space")
df["total_charges"] = pd.to_numeric(df.total_charges, errors="coerce")

section("7. Categoricals: spelling, case, whitespace")
print("plan before  :", df.plan.value_counts().to_dict())
print("region before:", df.region.value_counts().to_dict())
for c in ["plan", "region", "contract"]:
    df[c] = df[c].str.strip().str.lower()
print("plan after   :", df.plan.value_counts().to_dict())
print("region after :", df.region.value_counts().to_dict())

section("8. Summary statistics -> impossible values and outliers")
print(df[["age", "tenure_months", "monthly_charges", "support_calls"]].describe().round(1))
print("ages outside [18, 100]:", (~df.age.between(18, 100)).sum())
print("monthly_charges >= 1000:", (df.monthly_charges >= 1000).sum(),
      "| negative:", (df.monthly_charges < 0).sum())
df["age"] = df.age.where(df.age.between(18, 100))                 # impossible -> NaN
df["monthly_charges"] = df.monthly_charges.where(df.monthly_charges.between(0, 1000))

section("9. The target")
print(df.churned.value_counts(normalize=True).round(3).to_dict(),
      " -> imbalanced-ish; accuracy alone will mislead (week 4)")

section("10. Leakage hunt")
print(pd.crosstab(df.cancellation_reason.isna(), df.churned))
print("-> cancellation_reason is only known AFTER churn. Drop it before modelling!")
df = df.drop(columns="cancellation_reason")

section("11. Is missingness informative?")
print(df.groupby(df.support_calls.isna()).churned.mean().rename("churn rate")
      .rename_axis("support_calls missing").round(3))
print("-> yes: 'missing' itself predicts churn. Keep a missing-indicator feature (week 6).")

section("12. What moves churn? Group-bys and pivots")
print(df.groupby("contract").agg(churn_rate=("churned", "mean"), n=("churned", "size")).round(3))
print(df.pivot_table(index="plan", columns="contract", values="churned", aggfunc="mean").round(3))
df["tenure_bucket"] = pd.cut(df.tenure_months, [-1, 6, 12, 24, 48, 72],
                             labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
print(df.groupby("tenure_bucket", observed=True).churned.mean().round(3))
print("numeric correlations with churn:")
print(df.select_dtypes("number").corr()["churned"].drop("churned").sort_values().round(3))

section("13. Feature from a date")
df["signup_month"] = df.signup_date.dt.month
print(df.groupby("signup_month").churned.mean().round(3).head(4), "...")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
ax = axes[0, 0]
for label, g in df.groupby("churned"):
    ax.hist(g.tenure_months, bins=36, alpha=0.6, density=True, label=f"churned={label}")
ax.set(title="Tenure by churn", xlabel="months"); ax.legend()

ax = axes[0, 1]
df.boxplot(column="monthly_charges", by="plan", ax=ax)
ax.set(title="Monthly charges by plan", xlabel=""); fig.suptitle("")

ax = axes[0, 2]
rates = df.groupby("contract").churned.agg(["mean", "size"])
ax.bar(rates.index, rates["mean"])
for i, (r, n) in enumerate(zip(rates["mean"], rates["size"])):
    ax.text(i, r + 0.01, f"n={n}", ha="center")
ax.set(title="Churn rate by contract", ylabel="churn rate")

ax = axes[1, 0]
calls = df.groupby("support_calls").churned.mean()
ax.plot(calls.index, calls.values, "o-"); ax.set(title="Churn vs support calls", xlabel="calls")

ax = axes[1, 1]
corr = df.select_dtypes("number").corr()
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
fig.colorbar(im, ax=ax); ax.set_title("Correlation heatmap")

ax = axes[1, 2]
sample = df.sample(1500, random_state=0)
ax.scatter(sample.tenure_months, sample.total_charges, c=sample.churned, s=6, alpha=0.5, cmap="coolwarm")
ax.set(title="Total charges vs tenure (red = churned)", xlabel="tenure", ylabel="total charges")

fig.tight_layout(); fig.savefig(OUT / "eda.png", dpi=110)
print("\nsaved", OUT / "eda.png")
print("clean shape:", df.shape)
