# 5.5 Principal component analysis (PCA)

PCA is week 1's SVD put to work on data.

## The goal

Find a small number of directions (principal components) that capture as much
of the data's variance as possible, and represent each point by its
coordinates along them.

- **PC1** = the unit direction along which projected data has maximum variance.
- **PC2** = the max-variance direction orthogonal to PC1, and so on.

## Two equivalent derivations

1. **Maximise variance:** maximise $\text{Var}(Xv) = v^\top \Sigma v$ subject to $\lVert v\rVert = 1$, where Σ is the covariance matrix. Lagrange multipliers give $\Sigma v = \lambda v$ — **the eigenvectors of the covariance matrix**; the variance captured is the eigenvalue λ.
2. **Minimise reconstruction error:** find the k-dimensional subspace closest to the data in squared distance. By Eckart–Young (week 1.4), it's spanned by the top-k right singular vectors.

Computation, in practice:

```python
Xc = X - X.mean(0)                                   # 1. centre (always!)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)    # 2. SVD of centred data
components = Vt[:k]                                  # 3. directions (k, d)
Z = Xc @ components.T                                # 4. scores / projections (n, k)
explained_variance = S**2 / (n - 1)                  #    = eigenvalues of the covariance
ratio = S**2 / (S**2).sum()
X_back = Z @ components + X.mean(0)                  # 5. reconstruction
```

## Interpreting PCA

- **Explained variance ratio:** fraction of total variance per component. A
  **scree plot** (ratio vs component) and the cumulative curve help pick k
  (e.g. "keep 95 % of variance").
- **Loadings** (entries of each component): how much each original feature
  contributes. PC1 of standardised exam scores is often "general ability"
  (all positive loadings); PC2 a contrast ("maths vs languages").
- The **sign** of a component is arbitrary (v and −v are both valid).

## Standardise first? Usually yes

PCA maximises variance, so a feature measured in larger units dominates PC1 just
by scale. Standardise unless features share a natural common unit (e.g.
pixel intensities, or embeddings).

## Uses

| Use | Notes |
|---|---|
| **Visualisation** | project to 2-D to eyeball structure (t-SNE/UMAP are better for non-linear structure, but PCA is honest about distances) |
| **Compression / speed-up** | fewer features for downstream models; reduce embedding dimension before indexing |
| **De-correlation / denoising** | components are uncorrelated; dropping small ones removes noise |
| **Collinearity fix** | regression on principal components |
| **Analysing embeddings & LLM activations** | the top PCs of hidden states often reveal interpretable directions (Phase 3/7) |

## Limitations

- Linear only — curved manifolds need kernel PCA, autoencoders, UMAP.
- Maximum variance ≠ most predictive: a low-variance direction can carry the label. Never fit PCA and assume it helps a supervised model — validate it.
- Fit PCA on **training data only** (week 6 — leakage).

## Run the demo

```bash
python pca.py
```

PCA from scratch (both via covariance eigen-decomposition and via SVD, checked
against scikit-learn), explained variance on correlated data, the scaling
trap, a 64-pixel digits dataset compressed and reconstructed, a 2-D projection
plot, and a case where PCA throws away the label.

## Exercises

1. Show that the PC scores are uncorrelated (their covariance matrix is diagonal).
2. 100 features, the top 10 components explain 97 % of variance. What does that say about the data's rank, and what would you try?
3. Construct a 2-D dataset where PC1 is useless for classification.

<details><summary>Answers</summary>

1. $Z = X_cV$, so $\frac{1}{n-1}Z^\top Z = V^\top\Sigma V = \Lambda$ (diagonal), since V holds Σ's eigenvectors.
2. The data is approximately rank ~10 (many features are redundant/correlated). Try training on the 10 PC scores — faster, possibly less over-fitting — and compare on validation.
3. Two long, parallel, thin clusters side by side along x (large variance in x within each cluster) separated slightly in y: PC1 is x, but the label depends only on y. The demo builds exactly this.

</details>
