"""Generates a realistic, deliberately messy customer-churn dataset.

Problems planted on purpose (find them during EDA):
- inconsistent category spellings / case / whitespace in `plan` and `region`
- `total_charges` stored as text with some "N/A" / blank values
- missing values in `monthly_charges` and `support_calls` (support_calls missing more for churners)
- impossible ages (-1, 150) and a few extreme outliers in `monthly_charges`
- duplicated customer rows
- a leaky column, `cancellation_reason`, filled only for churned customers
"""
import numpy as np
import pandas as pd


def make_churn(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    plan = rng.choice(["basic", "standard", "premium"], n, p=[0.5, 0.35, 0.15])
    contract = rng.choice(["monthly", "one_year", "two_year"], n, p=[0.55, 0.25, 0.2])
    region = rng.choice(["north", "south", "east", "west"], n)
    age = rng.normal(40, 12, n).clip(18, 85).round()
    tenure = rng.exponential(24, n).clip(0, 72).round()
    base = {"basic": 25, "standard": 55, "premium": 95}
    monthly = np.array([base[p] for p in plan]) + rng.normal(0, 8, n)
    support_calls = rng.poisson(1.5, n)
    signup = pd.Timestamp("2026-09-30") - pd.to_timedelta(tenure * 30, unit="D")

    logit = (-1.0 + 1.2 * (contract == "monthly") - 0.9 * (contract == "two_year")
             - 0.04 * tenure + 0.35 * support_calls + 0.01 * (monthly - 50)
             - 0.4 * (plan == "premium") + rng.normal(0, 0.5, n))
    churned = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "signup_date": signup.strftime("%Y-%m-%d"),
        "age": age, "region": region, "plan": plan, "contract": contract,
        "tenure_months": tenure, "monthly_charges": monthly.round(2),
        "support_calls": support_calls.astype(float), "churned": churned,
    })
    df["total_charges"] = (df.monthly_charges * df.tenure_months).round(2).astype(str)
    df["cancellation_reason"] = np.where(
        churned == 1, rng.choice(["price", "service", "moved", "competitor"], n), None)

    # ---- make it messy ----
    messy = rng.uniform(size=n)
    df.loc[messy < 0.05, "plan"] = df.loc[messy < 0.05, "plan"].str.upper()
    df.loc[(messy >= 0.05) & (messy < 0.08), "plan"] = " " + df.plan + " "
    df.loc[rng.uniform(size=n) < 0.04, "region"] = "North"
    df.loc[rng.uniform(size=n) < 0.03, "total_charges"] = "N/A"
    df.loc[rng.uniform(size=n) < 0.01, "total_charges"] = " "
    df.loc[rng.uniform(size=n) < 0.06, "monthly_charges"] = np.nan
    miss_calls = rng.uniform(size=n) < np.where(churned == 1, 0.15, 0.03)
    df.loc[miss_calls, "support_calls"] = np.nan
    df.loc[rng.choice(n, 8, replace=False), "age"] = -1
    df.loc[rng.choice(n, 5, replace=False), "age"] = 150
    df.loc[rng.choice(n, 6, replace=False), "monthly_charges"] = 9999.0
    dupes = df.sample(60, random_state=seed)
    return pd.concat([df, dupes]).sample(frac=1, random_state=seed).reset_index(drop=True)


def clean_churn(df):
    """The fixes found during week 3's EDA, as one idempotent function.
    Drops the leaky `cancellation_reason` column. Does NOT impute (that belongs in a pipeline)."""
    df = df.drop_duplicates("customer_id").copy()
    for c in ["plan", "region", "contract"]:
        df[c] = df[c].astype(str).str.strip().str.lower()
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["age"] = df["age"].where(df["age"].between(18, 100))
    df["monthly_charges"] = df["monthly_charges"].where(df["monthly_charges"].between(0, 1000))
    return df.drop(columns=["cancellation_reason"], errors="ignore").reset_index(drop=True)


if __name__ == "__main__":
    raw = make_churn()
    clean = clean_churn(raw)
    print(raw.shape, "->", clean.shape)
    print(clean.head())
