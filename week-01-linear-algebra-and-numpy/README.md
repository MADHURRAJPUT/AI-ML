# Week 1 — Linear algebra & NumPy

> "Think of every layer as a matrix multiply." — this week makes that sentence obvious.

## Why this week matters

A neural network layer is `y = activation(W @ x + b)`. An embedding lookup is a
matrix row. Attention is three matrix multiplies and a softmax. A recommender is
a low-rank matrix factorisation. PCA is an SVD. If you are fluent in vectors and
matrices, roughly half of ML notation stops being intimidating.

## Learning goals

By Sunday you can:

1. Explain a dot product geometrically (projection, cosine similarity) and use it for search.
2. Read `(B, T, D) @ (D, H)` and say what shape comes out and what it means.
3. Explain rank, why a rank-deficient `XᵀX` breaks linear regression, and what to do.
4. Explain eigenvectors and SVD, and compress data/images with a truncated SVD.
5. Write NumPy with no Python loops over data, using broadcasting.

## Sub-folders (do them in order)

| # | Folder | Time | Core idea |
|---|---|---|---|
| 1 | [`01-vectors-and-dot-products`](01-vectors-and-dot-products) | 2.5 h | vectors as data points; dot product = similarity |
| 2 | [`02-matrix-multiplication`](02-matrix-multiplication) | 2.5 h | matmul = linear transformation = a layer |
| 3 | [`03-transpose-rank-and-inverse`](03-transpose-rank-and-inverse) | 2.5 h | what information a matrix keeps or destroys |
| 4 | [`04-eigenvectors-and-svd`](04-eigenvectors-and-svd) | 3 h | the directions a matrix cares about |
| 5 | [`05-numpy-vectorisation`](05-numpy-vectorisation) | 3 h | broadcasting, axis, einsum, speed |
| 6 | [`06-exercises`](06-exercises) | 3 h | 10 exercises + solutions |

Remaining ~2 h: reading and public output (below).

## Reading for the week

- *Mathematics for Machine Learning* (MML), ch. 2 (Linear Algebra) §2.1–2.7 and ch. 4 §4.2, 4.4–4.5 (eigen, SVD).
- 3Blue1Brown, *Essence of Linear Algebra*, episodes 1–4, 9, 13–14.

**Paper / blog for the 200-word summary:** Jay Alammar, *"The Illustrated Word2vec"* —
it is all dot products and is the bridge to embeddings in Phase 4.

## Public output idea

A short post: *"Cosine similarity, explained with 10 lines of NumPy"* using the
demo in `01-vectors-and-dot-products`.

## End-of-week self-check

- [ ] Without notes, what is the shape of `A @ B` for `A: (m, n)`, `B: (n, p)`? What's each output element?
- [ ] Why is cosine similarity preferred over raw dot product for comparing embeddings of different lengths?
- [ ] What does it mean if a matrix has rank lower than its number of columns, in data terms?
- [ ] What do the singular values of a data matrix tell you?
- [ ] Rewrite a double for-loop distance computation as one broadcasted NumPy expression.
