# 11.1 Embeddings: words, users and images as vectors

> Curriculum: *"Embeddings: representing words, images and users as vectors."*

An **embedding** maps a discrete thing (a word, a token, a user, a product) or
a complex thing (an image, a document) to a dense vector, learned so that
**geometry carries meaning**: similar things end up close together (week 1:
cosine similarity). Embeddings are the interface between the world and every
modern model — the first layer of every LLM, the index of every RAG system,
the core of every recommender.

## `nn.Embedding` is just a lookup table

```python
emb = nn.Embedding(num_embeddings=50_000, embedding_dim=768)   # a (50000, 768) weight matrix
vectors = emb(token_ids)                                       # (B, T) ints -> (B, T, 768)
```

Mathematically it's `one_hot(ids) @ W` — a linear layer on one-hot inputs —
but implemented as row indexing, so it's cheap. Gradients only flow into the
rows that were used (the demo shows this). In GPT-2 the token-embedding matrix
holds 38.6 M of the 124 M parameters.

## Three ways embeddings get learned

### 1. Words: from co-occurrence (word2vec)

*"You shall know a word by the company it keeps."* Skip-gram learns vectors so
that a word's vector predicts its neighbours. Negative sampling makes it cheap:
push up the score of a real (centre, context) pair, push down k random pairs:

$$\mathcal L = -\log\sigma(u_c\cdot v_o) - \sum_{k}\log\sigma(-u_c\cdot v_k)$$

In the demo, a toy corpus of animals, foods, vehicles and countries yields
vectors where `cat`'s neighbours are other animals. Modern LLMs learn token
embeddings the same way in spirit — as a side effect of predicting the next
token.

### 2. Users and items: from interactions (matrix factorisation)

Netflix-prize style: rating ≈ user vector · item vector + biases. With 10 %
of a ratings matrix observed, 8-number embeddings recover the hidden
structure and predict held-out ratings far better than the mean. Two-tower
retrieval models (one network embeds users, one embeds items) are the deep
version that powers YouTube/TikTok-style recommendations (Phase 6 system design).

### 3. Images (and documents): from a trained network's features

Take a network trained for some task, drop the classifier, and use the
penultimate activations as the embedding. The demo uses the week 10 CNN's
128-d global-pooled features: nearest neighbours mostly share the class →
image search for free. CLIP (Phase 3) trains image and text encoders jointly so
both live in **one** space — text-to-image search.

## Properties you'll rely on in Phase 4 (RAG)

- **Normalise** and use cosine/dot product (week 1.1).
- Dimension is a trade-off: 384–3072 for text embedding models; bigger = more expressive, more storage and slower search.
- Embeddings from *different* models aren't comparable — re-embed everything when you switch models.
- They encode biases present in their training data (word-vector analogies famously reproduce stereotypes).

## Run the demo

```bash
python embeddings.py      # the image part needs the week 10.3 CNN checkpoint (train_cnn.py)
```

## Exercises

1. Add a fifth group (e.g. colours) to the corpus and check its words cluster together.
2. In the matrix-factorisation demo, vary the embedding size (1, 3, 8, 32). Plot held-out RMSE. Where does it over-fit?
3. Use the image embeddings to find the 10 test images *farthest* from their class centroid. What are they?

<details><summary>Hints</summary>

2. k_true = 3 generated the data: 1 under-fits, ~3–8 is best, large sizes over-fit without stronger weight decay.
3. Compute class centroids of the normalised features; the outliers are often mislabelled or genuinely ambiguous images — a great way to find label noise.

</details>
