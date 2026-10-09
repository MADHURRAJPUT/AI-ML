# 1.6 Week 1 exercises

Ten exercises, from warm-up to "this is an interview question". Each one is a
function stub in `exercises.py`. Fill them in, then run:

```bash
python exercises.py      # checks YOUR implementations
python solutions.py      # checks the reference implementations (peek only after trying!)
```

Rules: **no Python loops over data** unless the exercise says so, and no
`sklearn`. NumPy only.

| # | Function | Skill |
|---|---|---|
| 1 | `cosine_similarity_matrix(A, B)` | normalisation + matmul |
| 2 | `project_onto(u, v)` | projection formula |
| 3 | `linear_layer(X, W, b, activation)` | shapes, broadcasting |
| 4 | `gram_schmidt(V)` | orthogonalisation (QR by hand) |
| 5 | `numerical_rank(X, tol)` | rank via singular values |
| 6 | `low_rank_approx(X, k)` | truncated SVD |
| 7 | `pca_fit_transform(X, k)` | PCA = SVD of centred data |
| 8 | `power_iteration(A, iters)` | top eigenpair |
| 9 | `knn_predict(X_train, y_train, X_test, k)` | pairwise distances + top-k + majority vote |
| 10 | `batched_attention_scores(Q, K)` | 4-D einsum + scaling + stable softmax |

## Stretch goals

- Time your `knn_predict` on 5 000 × 1 000 points. Make it 2× faster.
- Implement exercise 10 *without* einsum (using `transpose` and `@`).
- Write a 200-word blog post explaining exercise 7 (PCA) to a web developer.
