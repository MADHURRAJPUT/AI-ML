"""Phase 1 checkpoint — every whiteboard claim about logistic regression, verified numerically."""
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)


def section(title):
    print(f"\n=== {title} ===")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


N, d = 500, 3
X = rng.standard_normal((N, d))
w_true, b_true = np.array([1.5, -2.0, 0.5]), 0.3
y = (rng.uniform(size=N) < sigmoid(X @ w_true + b_true)).astype(float)

section("Claim 3: cross-entropy == negative log-likelihood of Bernoulli labels")
w, b = rng.standard_normal(d), 0.1
p = sigmoid(X @ w + b)
nll = -np.mean(np.log(np.where(y == 1, p, 1 - p)))
bce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
print(f"mean -log P(y|x) = {nll:.6f}   BCE = {bce:.6f}")

section("Claim 6: dℓ/dz = p − y  and  ∇w L = Xᵀ(p − y)/N")
z0, y0 = 0.7, 1.0
ell = lambda z: -(y0 * np.log(sigmoid(z)) + (1 - y0) * np.log(1 - sigmoid(z)))
num = (ell(z0 + 1e-6) - ell(z0 - 1e-6)) / 2e-6
print(f"dℓ/dz numeric={num:.6f}  p − y={sigmoid(z0) - y0:.6f}")


def loss(w, b):
    p = sigmoid(X @ w + b)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


grad_w = X.T @ (sigmoid(X @ w + b) - y) / N
num_w = np.array([(loss(w + e, b) - loss(w - e, b)) / 2e-6 for e in np.eye(d) * 1e-6])
print("∇w analytic:", grad_w.round(6))
print("∇w numeric :", num_w.round(6))

section("Claim 5: MSE-on-sigmoid is non-convex; cross-entropy is convex (1-D slice)")
X1 = rng.standard_normal(200); y1 = (X1 + rng.normal(0, 0.5, 200) > 0).astype(float)
ws = np.linspace(-15, 15, 601)
ce = np.array([np.mean(np.logaddexp(0, w_ * X1) - y1 * w_ * X1) for w_ in ws])
mse = np.array([np.mean((sigmoid(w_ * X1) - y1) ** 2) for w_ in ws])
second = lambda f: f[2:] - 2 * f[1:-1] + f[:-2]
print(f"min second difference: cross-entropy={second(ce).min():.2e} (≥ 0: convex)   "
      f"MSE={second(mse).min():.2e} (< 0: non-convex)")

section("Claim 5b: when confidently wrong, MSE's gradient vanishes, CE's doesn't")
for z in [-10, -5, 0]:
    p_ = sigmoid(z)
    print(f"z={z:4d} (y=1): dCE/dz={p_ - 1:+.4f}   dMSE/dz={2 * (p_ - 1) * p_ * (1 - p_):+.6f}")

section("Claim 8: gradient descent on the convex loss reaches the maximum-likelihood solution")
w, b = np.zeros(d), 0.0
for _ in range(3000):
    p = sigmoid(X @ w + b)
    w -= 0.5 * X.T @ (p - y) / N
    b -= 0.5 * np.mean(p - y)
mle = LogisticRegression(C=1e10, max_iter=10_000).fit(X, y)
print(f"our GD     : w={w.round(3)} b={b:.3f}")
print(f"exact MLE  : w={mle.coef_[0].round(3)} b={mle.intercept_[0]:.3f}")
print(f"true values: w={w_true} b={b_true}   (MLE ≠ truth with N=500: that's sampling noise, not a bug)")

section("Claim 7: softmax + cross-entropy gradient is Xᵀ(P − Y)/N")
K = 4
Yk = np.eye(K)[rng.integers(0, K, N)]
W = rng.standard_normal((d, K)) * 0.1


def softmax_loss(W):
    Z = X @ W
    Z = Z - Z.max(1, keepdims=True)
    P = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
    return -np.mean(np.sum(Yk * np.log(P), 1)), P


_, P = softmax_loss(W)
G = X.T @ (P - Yk) / N
E = np.zeros_like(W); E[1, 2] = 1e-6
print(f"dL/dW[1,2]: analytic={G[1, 2]:.6f} numeric={(softmax_loss(W + E)[0] - softmax_loss(W - E)[0]) / 2e-6:.6f}")
print("\nIf every line above matches, you can defend the whiteboard. On to Phase 2.")
