# 11.2 RNNs and LSTMs — and why they weren't enough

> Curriculum: *"RNNs and LSTMs as background for why transformers won."* You
> won't deploy an RNN in an LLM job — but you need to know what problem
> attention solved, and interviewers ask.

## The recurrent idea

Process a sequence one element at a time, carrying a **hidden state** that
summarises everything so far:

$$h_t = \tanh(W_{ih}x_t + W_{hh}h_{t-1} + b)$$

The same weights are reused at every step (weight sharing across *time*, as a
CNN shares across *space*). For language modelling, each $h_t$ feeds a
softmax over the vocabulary to predict token $t+1$ — the same next-token
cross-entropy objective as GPT.

### Training: backpropagation through time (BPTT)

Unroll the loop into a T-layer deep network with tied weights, then backprop.
The gradient reaching step 0 is a product of T Jacobians, each containing
$W_{hh}$ and tanh′ ≤ 1:

$$\frac{\partial h_T}{\partial h_0} = \prod_{t=1}^{T}\text{diag}(\tanh'(\cdot))\,W_{hh}$$

— week 1.4's repeated multiplication again. It **vanishes** (or explodes)
geometrically. The demo measures it: at initialisation the gradient at step 0 is
**10⁻¹⁸ ×** the gradient at step 49 for a vanilla RNN. Gradient clipping fixes
explosions; vanishing needs an architectural fix.

## LSTM: a gated, additive memory

The LSTM (Hochreiter & Schmidhuber, 1997) adds a **cell state** $c_t$ updated
*additively*, controlled by gates (sigmoids in [0, 1]):

$$
\begin{aligned}
f_t, i_t, o_t &= \sigma(\cdot),\ \sigma(\cdot),\ \sigma(\cdot) && \text{forget, input, output gates}\\
\tilde c_t &= \tanh(W_c[x_t, h_{t-1}] + b_c) && \text{candidate}\\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde c_t && \text{additive update}\\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

When $f_t \approx 1$, $c_t \approx c_{t-1}$ and the gradient flows back almost
unchanged — a **residual connection through time** (9.1). The demo: LSTM's
step-0 gradient is ~10⁻¹⁰× the last step's, vs ~10⁻¹⁸× for the RNN.
The GRU is a simpler two-gate variant with similar performance.

## What the demo shows

```bash
python rnn_lstm.py      # ~2-3 min on CPU; downloads names.txt once
```

**1. An RNN cell from scratch** matches `nn.RNN` exactly.

**2. Character-level name generation** (Karpathy's makemore data, 32k names), validation loss in nats per character:

| model | val loss |
|---|---|
| uniform guess | 3.296 (= ln 27) |
| bigram counts | 2.451 |
| RNN, 128 hidden | 2.072 |
| LSTM, 128 hidden | 2.008 |

Sampled LSTM names: *anzor, glema, avdel, xahri, jadel, brandyn, kestleigh,
adum* — mostly **not** in the training set: the model learned how names are
built, not a list. That's the same generative mechanism as an LLM, at the
character level.

**3. Long-range memory** — read T random digits, then output the *first* one:

| T | vanilla RNN | LSTM, default init | LSTM, forget-gate bias 2 |
|---|---|---|---|
| 10 | 0.93 | 1.00 | 1.00 |
| 20 | 0.47 | **0.10** (chance) | **1.00** |
| 50 | 0.09 | 0.10 | 0.41 |

Two honest lessons from this table:

- **The LSTM's advantage depends on initialisation.** With PyTorch's default (forget-gate bias 0, so the gate starts at σ(0) = 0.5) it fails at T = 20 — worse than the vanilla RNN. Initialising the forget-gate bias to a positive value so the cell starts out *remembering* (Jozefowicz et al., 2015) makes it solve T = 20 perfectly. Same theme as week 9: init matters.
- **Even a well-initialised LSTM struggles at T = 50** within this training budget. Information has to survive 50 sequential updates.

## Why this led to transformers

Two fundamental limits:

1. **Path length.** Information from step 1 reaches step T only through T − 1 updates. Long-range dependencies (a pronoun and its antecedent 200 tokens earlier) are hard to learn.
2. **Sequential computation.** $h_t$ needs $h_{t-1}$ — you can't parallelise across time. Training on billions of tokens is slow, and GPUs sit idle.

Attention (11.3) removes both: any position reads any other in one step, and
all positions are computed in parallel.

## Exercises

1. Add a GRU (`nn.GRU`) to the names comparison. Is it closer to the RNN or the LSTM?
2. Train the names LSTM with temperature 0.5 and 1.5 at sampling time. Describe the difference.
3. Find the forget-gate bias at which the LSTM first solves T = 20 (try 0, 0.5, 1, 2, 4).

<details><summary>Answers</summary>

1. Measure it: the GRU is often close to the LSTM in practice, but check on this data rather than assuming.
2. Low temperature: common, repetitive, "safe" names; high: more creative and more misspelled. The same knob as LLM sampling (Phase 3).
3. Run it — and note how sensitive the result is to one initialisation constant.

</details>
