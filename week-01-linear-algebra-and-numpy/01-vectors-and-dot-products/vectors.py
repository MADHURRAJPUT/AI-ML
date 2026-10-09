"""Week 1.1 — vectors, dot products, projection and cosine similarity."""
import numpy as np

rng = np.random.default_rng(0)


def section(title):
    print(f"\n=== {title} ===")


section("Dot product, two ways")
u = np.array([3.0, 4.0])
v = np.array([4.0, 3.0])
algebraic = sum(ui * vi for ui, vi in zip(u, v))
cos_theta = (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
geometric = np.linalg.norm(u) * np.linalg.norm(v) * cos_theta
print(f"u·v (loop)        = {algebraic}")
print(f"u·v (u @ v)       = {u @ v}")
print(f"|u||v|cosθ        = {geometric:.4f}")
print(f"angle between u,v = {np.degrees(np.arccos(cos_theta)):.2f}°")

section("Element-wise * vs dot product @")
print("u * v =", u * v, " <- a vector")
print("u @ v =", u @ v, "   <- a scalar")

section("Projection of u onto v")
proj = (u @ v) / (v @ v) * v
residual = u - proj
print("proj_v(u) =", proj)
print("residual  =", residual)
print("residual · v =", round(residual @ v, 10), "(orthogonal, as it should be)")

section("Toy semantic search with cosine similarity")
# Pretend 4-d embeddings: dimensions ~ [animal, vehicle, food, tech]
docs = {
    "cats are cute pets":         np.array([0.9, 0.0, 0.1, 0.0]),
    "dogs love walks":            np.array([0.8, 0.1, 0.0, 0.0]),
    "electric cars and batteries": np.array([0.0, 0.9, 0.0, 0.4]),
    "pizza recipe":               np.array([0.0, 0.0, 1.0, 0.0]),
    "GPUs for deep learning":     np.array([0.0, 0.05, 0.0, 3.0]),  # long vector!
}
query = np.array([0.1, 0.6, 0.0, 0.3])  # "tesla autopilot software"

names = list(docs)
D = np.stack([docs[n] for n in names])            # (5, 4)
dot_scores = D @ query                            # (5,)
D_unit = D / np.linalg.norm(D, axis=1, keepdims=True)
q_unit = query / np.linalg.norm(query)
cos_scores = D_unit @ q_unit

print(f"{'document':30s} {'dot':>7s} {'cosine':>7s}")
for i in np.argsort(-cos_scores):
    print(f"{names[i]:30s} {dot_scores[i]:7.3f} {cos_scores[i]:7.3f}")
print("Top by dot   :", names[np.argmax(dot_scores)])
print("Top by cosine:", names[np.argmax(cos_scores)])
print("-> the long 'GPUs' vector wins on raw dot product; cosine fixes that.")

section("Unit vectors: distance ranking == cosine ranking")
dist = np.linalg.norm(D_unit - q_unit, axis=1)
print("rank by distance:", np.argsort(dist))
print("rank by cosine  :", np.argsort(-cos_scores))

section("Why attention divides by sqrt(d)")
for d in [16, 64, 256, 1024]:
    q = rng.standard_normal((10_000, d))
    k = rng.standard_normal((10_000, d))
    scores = (q * k).sum(axis=1)
    print(f"d={d:5d}  std(q·k)={scores.std():7.2f}  sqrt(d)={np.sqrt(d):6.2f}  "
          f"std(q·k/sqrt(d))={(scores / np.sqrt(d)).std():.2f}")

section("Norms used in regularisation")
w = np.array([0.5, -2.0, 0.0, 1.5])
print("L1 =", np.abs(w).sum(), " L2 =", np.linalg.norm(w), " L2² =", w @ w)
