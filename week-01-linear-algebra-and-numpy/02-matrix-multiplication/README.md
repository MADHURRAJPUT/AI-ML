# 1.2 Matrix multiplication — the operation behind every layer

## Intuition first

A matrix is **a function that transforms vectors**. Multiplying `W @ x` takes a
vector `x` and produces a new vector. It can rotate, stretch, squash, shear or
project it — but it always keeps straight lines straight and the origin fixed.
That's what "linear" means.

A dense neural-network layer is exactly this, plus a shift and a non-linearity:

$$h = \sigma(Wx + b)$$

Stack a few and you have an MLP. Without σ, stacking is pointless: $W_2(W_1x) = (W_2W_1)x$ is still one linear map. **The non-linearity is what makes depth useful.**

## Shapes are the whole game

$$A \in \mathbb{R}^{m\times n},\quad B \in \mathbb{R}^{n\times p}\ \Rightarrow\ AB \in \mathbb{R}^{m\times p}$$

The inner dimensions must match and disappear. Each output entry is a dot product:

$$(AB)_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj} = \text{(row } i \text{ of } A) \cdot (\text{column } j \text{ of } B)$$

## Four ways to see `C = A @ B`

Being able to switch views is what makes papers readable.

1. **Dot products:** every `C[i, j]` is row `i` of A · column `j` of B.
2. **Column view:** every column of C is a *linear combination of the columns of A*, with weights from the matching column of B. → "`A @ x` is a weighted mix of A's columns."
3. **Row view:** every row of C is a linear combination of the rows of B. → "each data point (row) gets mapped into a new feature space."
4. **Sum of outer products:** $AB = \sum_k a_{:,k}\, b_{k,:}$ — a sum of rank-1 matrices. → This is the view behind SVD and LoRA (Phase 5).

## The batch convention in ML

In ML code data points are **rows**. A batch `X` has shape `(N, d_in)`. A layer
holds `W` of shape `(d_in, d_out)`:

```python
H = X @ W + b      # (N, d_in) @ (d_in, d_out) -> (N, d_out);  b: (d_out,) broadcast
```

PyTorch's `nn.Linear` stores `W` as `(d_out, d_in)` and computes `X @ W.T + b` —
same thing, transposed storage. In transformers you'll see 3-D tensors:

```python
X: (B, T, D)   # batch, tokens, model dim
W: (D, H)
X @ W -> (B, T, H)   # matmul applies to the last two dims, batch dims broadcast
```

## Properties worth memorising

| Property | True? | Why it matters |
|---|---|---|
| $(AB)C = A(BC)$ | ✅ associative | lets you choose the cheaper order (see below) |
| $AB = BA$ | ❌ not commutative | order of layers matters |
| $(AB)^\top = B^\top A^\top$ | ✅ | appears in every backprop derivation |
| $A(B + C) = AB + AC$ | ✅ distributive | |
| $I A = A$ | ✅ identity | residual connections are "`x + f(x)`" = `(I + f)` |

### Cost of a matmul

`(m, n) @ (n, p)` costs ≈ `2·m·n·p` FLOPs. That's the unit cost of deep learning;
GPU spec sheets quote it (TFLOPS). Associativity can save orders of magnitude:
for `x` a vector, `(A @ B) @ x` costs O(n³) but `A @ (B @ x)` costs O(n²).

## Common geometric transformations (2-D)

| Matrix | Effect |
|---|---|
| `[[s, 0], [0, s]]` | uniform scale |
| `[[cos t, -sin t], [sin t, cos t]]` | rotate by t |
| `[[1, k], [0, 1]]` | shear |
| `[[1, 0], [0, 0]]` | project onto x-axis (destroys information!) |

The demo applies each to a unit square and saves a plot.

## Run the demo

```bash
python matmul.py
```

Shows: a naive triple-loop matmul vs `@` (correctness and speed), the four views
verified numerically, a dense layer forward pass, 3-D batched matmul shapes,
the associativity speed trick, and 2-D transformations plotted to `outputs/`.

## Exercises

1. `X` is `(32, 128, 512)` and `W_q` is `(512, 64)`. Shape of `X @ W_q`? How many FLOPs (approximately)?
2. Show that two stacked linear layers (no activation) with widths 512→2048→512 are equivalent to a single 512×512 layer. What is the parameter count of each? (This is why we need activations — and also a hint at LoRA.)
3. Prove $(AB)^\top = B^\top A^\top$ using the index definition.

<details><summary>Answers</summary>

1. `(32, 128, 64)`. FLOPs ≈ 2 · (32·128) · 512 · 64 ≈ 268 M.
2. $W_2 W_1$ is 512×512. Two layers: 512·2048 + 2048·512 ≈ 2.1 M weights; one layer: 262 k. The product of the two can't express *more* than a 512×512 matrix can.
3. $((AB)^\top)_{ij} = (AB)_{ji} = \sum_k A_{jk}B_{ki} = \sum_k (B^\top)_{ik}(A^\top)_{kj} = (B^\top A^\top)_{ij}$.

</details>
