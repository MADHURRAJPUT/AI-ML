"""Feature engineering. Stateless row-wise features live here as plain functions;
anything that LEARNS from data (imputing, scaling, encoding) is built as a pipeline step
so cross-validation refits it on every training fold."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


# ---------------------------------------------------------------- churn ----
def churn_features(df):
    df = df.copy()
    df["charges_per_tenure_month"] = df.total_charges / df.tenure_months.clip(lower=1)
    df["is_new"] = (df.tenure_months <= 6).astype(int)
    df["calls_per_year"] = df.support_calls / (df.tenure_months.clip(lower=1) / 12)
    df["signup_month"] = df.signup_date.dt.month
    return df.drop(columns=["signup_date"])


# ------------------------------------------------------ spaceship titanic ----
SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def spaceship_features(df):
    """Point-in-time safe: every feature is known at boarding time.
    Group size is computed from the rows passed in; in train.py we pass train+test together
    (no labels involved), which is standard and leak-free for this competition."""
    df = df.copy()
    cabin = df["Cabin"].str.split("/", expand=True)
    df["Deck"], df["CabinNum"], df["Side"] = cabin[0], pd.to_numeric(cabin[1], errors="coerce"), cabin[2]
    df["GroupSize"] = df.groupby("Group")["PassengerId"].transform("count")
    df["Solo"] = (df["GroupSize"] == 1).astype(int)
    spend = df[SPEND]
    df["TotalSpend"] = spend.sum(axis=1, min_count=1)
    df["NoSpend"] = (spend.fillna(0).sum(axis=1) == 0).astype(int)
    df["LogTotalSpend"] = np.log1p(df["TotalSpend"])
    for c in SPEND:
        df[f"log_{c}"] = np.log1p(df[c])
    for c in ["CryoSleep", "VIP"]:
        df[c] = df[c].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    df["AgeBand"] = pd.cut(df["Age"], [-1, 12, 17, 25, 40, 60, 200], labels=False)
    return df.drop(columns=["Cabin", "Name", "Group"] + SPEND, errors="ignore")


FEATURE_FUNCS = {"churn": churn_features, "spaceship": spaceship_features}


def split_columns(df, exclude):
    cols = [c for c in df.columns if c not in exclude]
    categorical = [c for c in cols if df[c].dtype == object or str(df[c].dtype) in ("category", "str", "string")]
    numeric = [c for c in cols if c not in categorical]
    return numeric, categorical


def linear_preprocessor(numeric, categorical):
    """For logistic regression: impute (+ missing indicators), scale, one-hot."""
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numeric),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                              OneHotEncoder(handle_unknown="ignore", min_frequency=10)), categorical),
    ])


def tree_preprocessor(numeric, categorical):
    """For gradient boosting: leave numerics (NaN handled natively), ordinal-encode categoricals."""
    return ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                              OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
         categorical),
    ])
