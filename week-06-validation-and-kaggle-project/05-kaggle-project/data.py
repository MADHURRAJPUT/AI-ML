"""Dataset loading. Each loader returns a DatasetSpec so train.py stays dataset-agnostic."""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DatasetSpec:
    name: str
    train: pd.DataFrame
    test: Optional[pd.DataFrame]          # unlabelled rows to predict (Kaggle test.csv), if any
    target: str
    id_col: str
    group_col: Optional[str] = None       # rows sharing a group must stay in the same CV fold
    metric: str = "roc_auc"               # the competition / business metric
    drop: list = field(default_factory=list)   # columns never used as features


def load_churn(seed=0):
    """Offline default: the messy churn data from week 3, cleaned with common.clean_churn."""
    sys.path.insert(0, str(ROOT))
    from common.churn_data import clean_churn, make_churn

    df = clean_churn(make_churn(n=8000, seed=seed))
    return DatasetSpec(name="churn", train=df, test=None, target="churned",
                       id_col="customer_id", metric="roc_auc")


def load_spaceship(data_dir):
    """Kaggle 'Spaceship Titanic' (https://www.kaggle.com/competitions/spaceship-titanic).

    Download with:  kaggle competitions download -c spaceship-titanic -p data/spaceship && unzip ...
    """
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test_path = data_dir / "test.csv"
    test = pd.read_csv(test_path) if test_path.exists() else None
    train["Transported"] = train["Transported"].astype(int)
    for df in [train] + ([test] if test is not None else []):
        df["Group"] = df["PassengerId"].str.split("_").str[0]     # travel group (families etc.)
    return DatasetSpec(name="spaceship", train=train, test=test, target="Transported",
                       id_col="PassengerId", group_col="Group", metric="accuracy",
                       drop=["Name"])


def load_csv(path, target, id_col, group_col=None, metric="roc_auc"):
    """Any binary-classification CSV."""
    df = pd.read_csv(path)
    return DatasetSpec(name=Path(path).stem, train=df, test=None, target=target, id_col=id_col,
                       group_col=group_col, metric=metric)
