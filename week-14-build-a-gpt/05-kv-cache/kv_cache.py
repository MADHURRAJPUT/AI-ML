"""Week 14.5 — the KV cache: same outputs, much less work per generated token."""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.gpt import GPT, GPTConfig  # noqa: E402

torch.manual_seed(0)


def section(title):
    print(f"\n=== {title} ===")


section("Why: without a cache, every step recomputes keys and values for the whole prefix")
print("step t of naive generation runs the model on all t tokens, only to use the LAST position's logits.")
print("But k and v for tokens 0..t-1 are the same as last step (the causal mask means later tokens never")
print("change earlier ones). So store them: each step then processes ONE token and appends its k, v.")

section("Correctness: cached and uncached generation give identical tokens")
ckpt = ROOT / "runs" / "char_gpt.pt"
if ckpt.exists():
    model, _ = GPT.load(ckpt)
    print("using the trained char GPT from 14.2")
else:
    model = GPT(GPTConfig(vocab_size=65, block_size=128, n_layer=4, n_head=4, n_embd=128))
    print("(runs/char_gpt.pt not found; using a randomly initialised model of the same shape)")
model.eval()
start = torch.zeros(1, 1, dtype=torch.long)
a = model.generate(start, 100, temperature=0, use_cache=True)
b = model.generate(start, 100, temperature=0, use_cache=False)
print("greedy, 100 tokens: identical =", torch.equal(a, b))
ga, gb = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
a = model.generate(start, 100, temperature=1.0, use_cache=True, generator=ga)
b = model.generate(start, 100, temperature=1.0, use_cache=False, generator=gb)
print("sampled with the same seed: identical =", torch.equal(a, b))

section("Speed: a GPT-2-small-shaped model (12 layers, C=768; random weights — speed doesn't depend on them)")
big = GPT(GPTConfig(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768)).eval()
prompt = torch.randint(50257, (1, 16))
big.generate(prompt, 4, use_cache=True); big.generate(prompt, 4, use_cache=False)   # warm-up
for n_new in [64, 256]:
    times = {}
    for use_cache in [False, True]:
        t0 = time.perf_counter()
        big.generate(prompt, n_new, temperature=0, use_cache=use_cache)
        times[use_cache] = time.perf_counter() - t0
    print(f"{n_new:3d} new tokens: no cache {times[False]:6.2f} s | KV cache {times[True]:6.2f} s | "
          f"speed-up {times[False] / times[True]:4.1f}x")
print("-> the speed-up grows with length: uncached work per step grows with t, cached work stays nearly flat.")

section("Per-step latency as the sequence grows (same model)")
with torch.no_grad():
    for t in [32, 256, 768]:
        ctx = torch.randint(50257, (1, t))
        _, _, past = big(ctx[:, :-1], use_cache=True)
        reps = 3
        t0 = time.perf_counter()
        for _ in range(reps):
            big(ctx)                                             # no cache: process all t tokens
        full = (time.perf_counter() - t0) / reps
        t0 = time.perf_counter()
        for _ in range(reps):
            big(ctx[:, -1:], past_kv=past, use_cache=True)       # cache: process 1 token
        cached = (time.perf_counter() - t0) / reps
        print(f"context {t:3d}: one step without cache {1e3 * full:6.1f} ms | with cache {1e3 * cached:5.1f} ms")

section("The price: memory. KV cache = 2 (K,V) x layers x heads x head_dim x tokens x bytes")
for name, L, kvh, d, T, nbytes in [("our char GPT (fp32)", 4, 4, 32, 128, 4),
                                   ("GPT-2 small, 1 024 tokens (fp32)", 12, 12, 64, 1024, 4),
                                   ("LLaMA-2-7B, 4 096 tokens (fp16, MHA)", 32, 32, 128, 4096, 2),
                                   ("LLaMA-3-8B, 8 192 tokens (bf16, GQA 8)", 32, 8, 128, 8192, 2)]:
    b = 2 * L * kvh * d * T * nbytes
    print(f"{name:40s} {b / 2**20:10.1f} MiB per sequence")
print("-> multiply by the batch size. For long contexts the cache, not the weights, limits how many users a")
print("   GPU can serve at once: hence GQA (13.2), cache quantisation and PagedAttention/vLLM (week 16).")
