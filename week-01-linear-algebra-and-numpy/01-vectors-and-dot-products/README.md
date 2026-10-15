# 1.1 Vectors and dot products

## Intuition first

In ML, a **vector is a data point described by numbers**.

- A house: `[sqft, bedrooms, age]` → a vector in ℝ³.
- A word embedding from an LLM: a vector in ℝ⁷⁶⁸ or ℝ⁴⁰⁹⁶.
- A 28×28 grayscale image: flattened, a vector in ℝ⁷⁸⁴.

Two views of the same object — keep both in your head:

1. **A point** in space (where the data sits).
2. **An arrow** from the origin (a direction and a length).

## The operations you need

| Operation | Formula | NumPy | Meaning |
|---|---|---|---|
| Addition | $(u+v)_i = u_i + v_i$ | `u + v` | combine / shift |
| Scaling | $(cv)_i = c\,v_i$ | `c * v` | stretch / flip |
| Dot product | $u\cdot v = \sum_i u_i v_i$ | `u @ v` | alignment × lengths |
| L2 norm | $\lVert v\rVert_2 = \sqrt{v\cdot v}$ | `np.linalg.norm(v)` | length |
| L1 norm | $\lVert v\rVert_1 = \sum_i \lvert v_i\rvert$ | `np.abs(v).sum()` | "taxicab" length |

## The dot product, two ways

**Algebraic:** multiply element-wise, then sum.

$$u \cdot v = \sum_{i=1}^{n} u_i v_i$$

**Geometric:**

$$u \cdot v = \lVert u\rVert\,\lVert v\rVert \cos\theta$$

where θ is the angle between the arrows. So the dot product is large and positive
when vectors point the same way, zero when perpendicular (**orthogonal**), and
negative when they point in opposite directions.

### Projection

The length of the "shadow" of `u` onto the direction of `v` is
$\dfrac{u\cdot v}{\lVert v\rVert}$. The projected vector itself is

$$\text{proj}_v(u) = \frac{u\cdot v}{v\cdot v}\, v$$

Projection is the heart of linear regression (the prediction is the projection of
the target onto the span of the features — you'll see this in week 4) and of PCA
(project data onto the top directions — week 5).

### Cosine similarity

Divide out the lengths and only the angle remains:

$$\cos(u, v) = \frac{u \cdot v}{\lVert u\rVert\,\lVert v\rVert} \in [-1, 1]$$

This is **the** similarity function of semantic search and RAG (Phase 4). If all
vectors are normalised to length 1 first, cosine similarity *is* the dot product
— which is why vector databases often store normalised embeddings and use a
fast inner-product index.

## Where this shows up in ML

- **A neuron** computes `w · x + b` — a dot product between weights and input, i.e. "how aligned is the input with the pattern this neuron looks for".
- **Linear models:** prediction $\hat y = w\cdot x + b$.
- **Attention (Phase 3):** score between a query and a key is `q · k / sqrt(d)`.
- **Retrieval (Phase 4):** nearest neighbours by cosine similarity.
- **Regularisation (week 4):** L2 penalty $\lVert w\rVert_2^2$, L1 penalty $\lVert w\rVert_1$.

## Why divide by √d in attention? (a preview)

If `q` and `k` have independent entries with mean 0 and variance 1, then
`q · k` is a sum of `d` terms each with variance 1, so its variance is `d` and its
standard deviation is √d. Dividing by √d keeps the scores at unit scale so the
softmax doesn't saturate. You can verify this numerically in `vectors.py`.

## Common pitfalls

- **Comparing un-normalised embeddings with dot product** — longer vectors win regardless of meaning.
- **Mixing up `*` and `@`** — `u * v` is element-wise (a vector), `u @ v` is the dot product (a scalar).
- **Features on wildly different scales** (sqft ≈ 2000, bedrooms ≈ 3) dominate distances and dot products. Standardise first (week 6).

## Run the demo

```bash
python vectors.py
```

It shows: dot products both ways, projection, a tiny "semantic search" with
cosine similarity, and an empirical check of the √d scaling.

## Exercises

1. Prove that $\lVert u - v\rVert^2 = \lVert u\rVert^2 + \lVert v\rVert^2 - 2\,u\cdot v$. Why does this mean that, for unit vectors, *ranking by Euclidean distance is the same as ranking by cosine similarity*?
2. Find a non-zero vector orthogonal to `[1, 2, 3]`.
3. In `vectors.py`, change the query in the toy search. Can you construct a query whose dot-product ranking differs from its cosine ranking? What property of the document vectors makes that possible?

<details><summary>Answers</summary>

1. Expand $(u-v)\cdot(u-v)$. For unit vectors, $\lVert u-v\rVert^2 = 2 - 2\cos(u,v)$, a strictly decreasing function of cosine, so the ordering is identical.
2. Any `v` with `v1 + 2 v2 + 3 v3 = 0`, e.g. `[3, 0, -1]` or `[2, -1, 0]`.
3. Yes — when documents have different norms. A long vector that is only moderately aligned can have a bigger dot product than a short, perfectly aligned one.

</details>
