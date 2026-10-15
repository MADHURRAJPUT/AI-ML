"""Week 15.1 — pretraining at small scale: a mini scaling law on Shakespeare, and Chinchilla arithmetic."""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.gpt import GPT, GPTConfig  # noqa: E402
from common.lm_train import train  # noqa: E402
from common.text_data import CharVocab, load_text  # noqa: E402

OUT = Path(__file__).parent / "outputs"


def section(title):
    print(f"\n=== {title} ===")


section("A mini scaling law: same data, same steps, five model sizes (≈ 4 min)")
text = load_text("shakespeare")
vocab = CharVocab(text)
data = torch.tensor(vocab.encode(text))
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

STEPS, BATCH, BLOCK = 800, 32, 64
sizes = [(1, 2, 16), (2, 2, 32), (2, 4, 64), (3, 4, 96), (4, 4, 128)]
rows = []
for L, H, C in sizes:
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=len(vocab), block_size=BLOCK, n_layer=L, n_head=H, n_embd=C))
    N = model.num_params()
    t0 = time.time()
    hist = train(model, train_data, val_data, max_steps=STEPS, batch_size=BATCH, lr=2e-3, warmup=50,
                 eval_every=STEPS, log=lambda *_: None)
    D = STEPS * BATCH * BLOCK
    rows.append((N, hist[-1]["val"], 6 * N * D))
    last_secs = time.time() - t0
    print(f"L={L} C={C:3d}: N = {N:>8,} params | val loss {hist[-1]['val']:.3f} | compute 6ND = "
          f"{6 * N * D:.1e} FLOPs | {time.time() - t0:.0f} s")

N_arr, loss_arr = np.array([r[0] for r in rows], float), np.array([r[1] for r in rows])
slope, intercept = np.polyfit(np.log(N_arr), np.log(loss_arr), 1)
print(f"\nfit: loss ≈ {math.exp(intercept):.2f} · N^({slope:.3f})")
print("-> a straight line in log-log space: each 10x in parameters lowers the loss by a predictable factor.")
print(f"   (this tiny fit: x{10 ** slope:.2f} per 10x params. Real laws include an irreducible term, L(N) = E + A/N^α,")
print("   and are fitted over many orders of magnitude; ours spans ~2.)")

section("Chinchilla: how to spend a compute budget")
print("Training compute C ≈ 6·N·D FLOPs (2 for the forward pass + 4 for backward, per parameter per token).")
print("Hoffmann et al. 2022: for a fixed C, loss is minimised with D ≈ 20·N tokens.")
print(f"{'compute (FLOPs)':>16s} {'optimal N':>12s} {'optimal D (tokens)':>20s}")
for C in [1e18, 1e21, 5.8e23, 1e25]:
    N_opt = math.sqrt(C / (6 * 20))
    print(f"{C:16.1e} {N_opt / 1e9:10.2f} B {20 * N_opt / 1e9:17.0f} B")
print("(5.8e23 ≈ Chinchilla itself: 70 B params on 1.4 T tokens; it beat the 280 B Gopher trained on 300 B tokens.)")

section("...and why modern models ignore it on purpose")
for name, N, D in [("GPT-3 (2020)", 175e9, 300e9), ("Chinchilla (2022)", 70e9, 1.4e12),
                   ("LLaMA-2-7B (2023)", 7e9, 2e12), ("LLaMA-3-8B (2024)", 8e9, 15e12)]:
    print(f"{name:20s} {N / 1e9:5.0f} B params, {D / 1e12:5.2f} T tokens -> {D / N:6.0f} tokens/param, "
          f"compute {6 * N * D:.1e} FLOPs")
print("-> Chinchilla minimises TRAINING compute. But a model is trained once and served billions of times:")
print("   a smaller model trained far past 20 tokens/param costs more to train and much less to run.")

section("Turning FLOPs into wall-clock time")
N_big, C_big, secs = rows[-1][0], rows[-1][2], last_secs
print(f"measured here: the {N_big:,}-param model used {C_big:.1e} FLOPs in {secs:.0f} s "
      f"-> {C_big / secs:.1e} FLOP/s achieved on this CPU")
print(f"Chinchilla-optimal for it: 20·N = {20 * N_big / 1e6:.0f} M tokens -> {6 * N_big * 20 * N_big:.1e} FLOPs "
      f"-> ~{6 * N_big * 20 * N_big / (C_big / secs) / 60:.0f} min here")
print(f"GPT-3 (3.1e23 FLOPs) on 1 000 A100s (312 TFLOP/s bf16 peak, ~40% utilisation): "
      f"{3.1e23 / (1000 * 312e12 * 0.4) / 86400:.0f} days")
print(f"...and on this CPU: {3.1e23 / (C_big / secs) / 3.15e7:,.0f} years")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(N_arr, loss_arr, "o", label="measured")
    xs = np.logspace(np.log10(N_arr.min()), np.log10(N_arr.max()), 50)
    ax.loglog(xs, np.exp(intercept) * xs ** slope, "--", label=f"fit N^{slope:.3f}")
    ax.set(xlabel="parameters", ylabel="val loss", title="mini scaling law"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "scaling.png", dpi=110)
    print("\nsaved", OUT / "scaling.png")
except ImportError:
    pass
