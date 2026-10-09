"""Week 3.2 — Bayes' rule: base rates, sequential updating, Naive Bayes from scratch."""
import numpy as np

rng = np.random.default_rng(0)


def section(title):
    print(f"\n=== {title} ===")


def posterior(prior, sensitivity, specificity):
    p_pos = sensitivity * prior + (1 - specificity) * (1 - prior)
    return sensitivity * prior / p_pos


section("Medical test: analytic vs simulated")
prev, sens, spec = 0.01, 0.99, 0.95
print(f"analytic  P(sick | +) = {posterior(prev, sens, spec):.4f}")
n = 1_000_000
sick = rng.uniform(size=n) < prev
positive = np.where(sick, rng.uniform(size=n) < sens, rng.uniform(size=n) > spec)
print(f"simulated P(sick | +) = {sick[positive].mean():.4f}  "
      f"({positive.sum():,} positives, only {(sick & positive).sum():,} truly sick)")

section("Same test, different base rates (= same model, different deployment)")
for prev in [0.5, 0.1, 0.01, 0.001]:
    print(f"prevalence {prev:6.1%} -> precision P(sick|+) = {posterior(prev, sens, spec):6.1%}")

section("Sequential updating of a coin's bias (Beta prior)")
# Beta(a, b) prior on p; after h heads and t tails the posterior is Beta(a+h, b+t)
true_p = 0.7
a, b = 1, 1                             # uniform prior
flips = rng.uniform(size=50) < true_p
for i, f in enumerate(flips, 1):
    a, b = a + f, b + (not f)
    if i in (1, 5, 10, 50):
        mean = a / (a + b)
        sd = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        print(f"after {i:2d} flips: posterior mean {mean:.3f} ± {sd:.3f}")


# ---------------------------------------------------------- Naive Bayes ----
spam_words = ["free", "win", "money", "click", "offer", "prize", "cash", "now"]
ham_words = ["meeting", "project", "lunch", "report", "team", "schedule", "code", "review"]
common = ["the", "a", "to", "you", "and", "is", "for", "now"]
vocab = sorted(set(spam_words + ham_words + common))
w2i = {w: i for i, w in enumerate(vocab)}


def make_doc(is_spam):
    topical = spam_words if is_spam else ham_words
    other = ham_words if is_spam else spam_words
    words = (list(rng.choice(topical, rng.integers(2, 6)))
             + list(rng.choice(common, rng.integers(3, 8)))
             + list(rng.choice(other, rng.integers(0, 2))))     # some noise
    return words


def featurise(docs):
    X = np.zeros((len(docs), len(vocab)), dtype=int)
    for i, d in enumerate(docs):
        for w in d:
            X[i, w2i[w]] += 1
    return X


y = (rng.uniform(size=600) < 0.4).astype(int)       # 40% spam
X = featurise([make_doc(s) for s in y])
X_tr, X_te, y_tr, y_te = X[:400], X[400:], y[:400], y[400:]


class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.log_prior = np.log(np.bincount(y) / len(y))
        counts = np.stack([X[y == c].sum(0) for c in self.classes])        # (C, V)
        smoothed = counts + self.alpha
        self.log_lik = np.log(smoothed / smoothed.sum(1, keepdims=True))   # log P(w | c)
        return self

    def predict_log_proba(self, X):
        joint = X @ self.log_lik.T + self.log_prior        # log P(c) + Σ count·log P(w|c)
        return joint - np.logaddexp.reduce(joint, axis=1, keepdims=True)  # normalise

    def predict(self, X):
        return self.classes[self.predict_log_proba(X).argmax(1)]


section("Multinomial Naive Bayes spam filter from scratch")
nb = MultinomialNB().fit(X_tr, y_tr)
print(f"test accuracy (ours)   : {(nb.predict(X_te) == y_te).mean():.3f}")
try:
    from sklearn.naive_bayes import MultinomialNB as SkNB

    sk = SkNB(alpha=1.0).fit(X_tr, y_tr)
    print(f"test accuracy (sklearn): {(sk.predict(X_te) == y_te).mean():.3f}")
    print("probabilities match sklearn:",
          np.allclose(np.exp(nb.predict_log_proba(X_te)), sk.predict_proba(X_te)))
except ImportError:
    pass
top_spam = np.argsort(nb.log_lik[1] - nb.log_lik[0])[::-1][:5]
print("most spam-indicative words:", [vocab[i] for i in top_spam])

section("Why log-space? Multiplying many small probabilities underflows")
p = np.full(500, 0.01)
print(f"product of 500 probs of 0.01 = {np.prod(p)}   sum of logs = {np.log(p).sum():.1f}")

section("Why smoothing? A word never seen in a class")
nb0 = MultinomialNB(alpha=1e-12).fit(X_tr[:20], y_tr[:20])
print("with alpha≈0, some log P(w|c) =", nb0.log_lik.min().round(1),
      "-> one unseen word can veto a class")
