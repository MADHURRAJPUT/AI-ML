# 5.4 k-means clustering

## The objective

Partition n points into k clusters, each with a centre $\mu_j$, minimising the
**within-cluster sum of squares** (inertia):

$$J = \sum_{i=1}^n \lVert x_i - \mu_{c(i)}\rVert^2$$

## Lloyd's algorithm

```
initialise k centres
repeat until assignments stop changing:
    assign:  c(i) = argmin_j ||x_i − μ_j||²            (nearest centre)
    update:  μ_j = mean of points assigned to j          (centroid)
```

Each step can only decrease J (assignment picks the closest centre; the mean is
the point minimising squared distance to a set — week 3's "MSE → mean"). So it
always converges — but to a **local** minimum that depends on initialisation.

### k-means++ initialisation

Pick the first centre uniformly at random; pick each next centre with
probability proportional to $D(x)^2$ — the squared distance to the nearest centre
already chosen. Spread-out starts → much better local minima, provably within
O(log k) of optimal in expectation. It's the default in scikit-learn. Also run
several restarts (`n_init`) and keep the lowest J.

## Choosing k

- **Elbow method:** plot J vs k; look for the bend (J always decreases with k, so you can't just minimise it).
- **Silhouette score:** for each point, $s = (b - a)/\max(a, b)$ with a = mean distance to its own cluster, b = mean distance to the nearest other cluster. Ranges −1…1; pick k with the highest average.
- **Downstream usefulness:** most often in business, k is chosen by what the segments are *for* (e.g. 4–6 marketing personas).

## Assumptions and failure modes

k-means implicitly assumes clusters are **spherical, similar-sized and similar-density**, since it uses Euclidean distance to a centre. It fails on:

- elongated or non-convex clusters (two moons, rings) → use DBSCAN, spectral clustering, or a Gaussian mixture;
- very different cluster sizes/densities;
- **unscaled features** — a feature in thousands dominates the distance. Standardise first;
- high-dimensional raw data (distances concentrate) → reduce with PCA first, or cluster *embeddings*.

## Where you'll use it as an AI engineer

- Clustering **text embeddings** to discover topics in support tickets or user queries (Phase 4).
- Picking diverse examples for an eval set or for few-shot prompts.
- **Vector quantisation:** IVF indexes in vector databases (FAISS, pgvector `ivfflat`) run k-means over embeddings and search only the nearest clusters — that's how approximate nearest-neighbour search gets fast (Phase 4).
- Product quantisation in compressed embeddings; colour quantisation in images.

## Run the demo

```bash
python kmeans.py
```

From-scratch k-means with random and k-means++ init (vectorised assignment
using week 1's distance trick), multiple restarts, elbow + silhouette for
choosing k, the scaling trap, and the two-moons failure case. Plots in `outputs/`.

## Exercises

1. Prove the update step is optimal: $\arg\min_\mu\sum_i\lVert x_i-\mu\rVert^2 = \bar x$.
2. Why can k-means never increase J during an iteration? Why can it still end in a bad solution?
3. You cluster customers on `[annual_spend_usd, num_visits]` without scaling. What happens?

<details><summary>Answers</summary>

1. Gradient $-2\sum(x_i - \mu) = 0$ ⇒ $\mu = \frac1n\sum x_i$.
2. Both steps minimise J over one set of variables holding the other fixed (coordinate descent). It's non-convex, so coordinate descent can stop at a local minimum determined by the start.
3. Distances are almost entirely driven by spend (thousands) not visits (tens); clusters become spend bands, ignoring visits. Standardise.

</details>
