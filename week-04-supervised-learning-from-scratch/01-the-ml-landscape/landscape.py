"""Week 4.1 — one tiny example of each task type, always compared with a baseline."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)


def section(title):
    print(f"\n=== {title} ===")


section("Regression: delivery minutes from distance and prep time")
X = np.column_stack([rng.uniform(0.5, 10, 2000), rng.uniform(5, 30, 2000)])
y = 8 + 3.5 * X[:, 0] + 0.9 * X[:, 1] + rng.normal(0, 4, 2000)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
for name, m in [("baseline: predict mean", DummyRegressor()), ("linear regression", LinearRegression())]:
    print(f"{name:25s} MAE = {mean_absolute_error(yte, m.fit(Xtr, ytr).predict(Xte)):.2f} min")

section("Binary classification on 95:5 data — the accuracy trap")
X = rng.standard_normal((4000, 5))
y = (X @ np.array([1.5, -1, 0.5, 0, 0]) + rng.normal(0, 1, 4000) > 3.5).astype(int)
print(f"positive rate = {y.mean():.3f}")
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0, stratify=y)
for name, m in [("baseline: majority class", DummyClassifier()),
                ("logistic regression", LogisticRegression())]:
    pred = m.fit(Xtr, ytr).predict(Xte)
    print(f"{name:25s} accuracy = {accuracy_score(yte, pred):.3f}   "
          f"positives found = {(pred[yte == 1] == 1).sum()}/{(yte == 1).sum()}")
print("-> the baseline scores ~95% accuracy while finding zero positives. Week 4.4 fixes the metric.")

section("Multi-class: 4 intents from 2-D 'embeddings'")
centres = np.array([[0, 0], [4, 0], [0, 4], [4, 4]])
y = rng.integers(0, 4, 2000)
X = centres[y] + rng.standard_normal((2000, 2))
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
for name, m in [("baseline: majority class", DummyClassifier()),
                ("softmax regression", LogisticRegression())]:
    print(f"{name:25s} accuracy = {accuracy_score(yte, m.fit(Xtr, ytr).predict(Xte)):.3f}")

section("Multi-label: K independent binary problems")
X = rng.standard_normal((1000, 3))
Y = np.column_stack([(X[:, 0] > 0), (X[:, 1] > 0.5), (X[:, 0] + X[:, 2] > 1)]).astype(int)
print("label combinations seen:", len({tuple(r) for r in Y}), "(a row can have several tags)")
for k, tag in enumerate(["billing", "bug", "urgent"]):
    acc = LogisticRegression().fit(X[:800], Y[:800, k]).score(X[800:], Y[800:, k])
    print(f"  tag {tag:8s} accuracy = {acc:.3f}")

section("Unsupervised: clustering without labels")
km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(centres[rng.integers(0, 4, 500)]
                                                          + rng.standard_normal((500, 2)) * 0.6)
print("cluster centres found:\n", km.cluster_centers_.round(2))
