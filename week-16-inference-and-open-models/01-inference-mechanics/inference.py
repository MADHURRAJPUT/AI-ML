"""Week 16.1 — inference mechanics: prefill vs decode, batching, quantisation, speculative decoding."""
import copy
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.gpt import GPT, GPTConfig  # noqa: E402
from common.lm_train import estimate_loss, train  # noqa: E402
from common.text_data import CharVocab, load_text  # noqa: E402

torch.manual_seed(0)
CKPT = ROOT / "runs" / "char_gpt.pt"
if not CKPT.exists():
    sys.exit(f"{CKPT} not found — run week-14-build-a-gpt/02-train-a-char-gpt/train_char.py first.")


def section(title):
    print(f"\n=== {title} ===")


section("Back-of-envelope: why decoding is memory-bound (7 B model, bf16, one H100)")
params, bytes_per = 7e9, 2
bw, flops = 3.35e12, 989e12                        # H100 SXM: HBM bandwidth (B/s), dense bf16 FLOP/s
t_mem = params * bytes_per / bw
for batch in [1, 8, 64, 256]:
    t_compute = 2 * params * batch / flops          # ~2 FLOPs per parameter per token
    t = max(t_mem, t_compute)
    print(f"batch {batch:3d}: read weights {1e3 * t_mem:5.2f} ms, compute {1e3 * t_compute:5.2f} ms per step -> "
          f"~{batch / t:8,.0f} tokens/s ({'memory' if t_mem > t_compute else 'compute'}-bound)")
print("-> each decode step must stream all 14 GB of weights from memory to produce ONE token per sequence.")
print("   Batching shares that read across sequences: throughput rises almost for free until compute catches up")
print("   (~batch 300 here). Prefill processes the whole prompt at once, so it is compute-bound from the start.")
print("   (The KV cache adds reads that grow with context length; ignored here.)")

section("Measured on this CPU: batching raises throughput (GPT-2-small shape, 32 new tokens, KV cache)")
big = GPT(GPTConfig(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768)).eval()
big.generate(torch.randint(50257, (1, 8)), 2)    # warm-up
for batch in [1, 4, 16]:
    prompt = torch.randint(50257, (batch, 8))
    t0 = time.perf_counter()
    big.generate(prompt, 32, temperature=0)
    dt = time.perf_counter() - t0
    print(f"batch {batch:2d}: {dt:5.2f} s -> {batch * 32 / dt:6.1f} tokens/s total, {dt / 32 * 1e3:6.1f} ms per step")
del big

model, ckpt = GPT.load(CKPT)
model.eval()
vocab = CharVocab("".join(ckpt["chars"]))
text = load_text("shakespeare")
data = torch.tensor(vocab.encode(text))
train_data, val_data = data[:int(0.9 * len(data))], data[int(0.9 * len(data)):]


def val_loss(m):
    return estimate_loss(m, {"val": val_data}, 128, 32, iters=20)["val"]


def quantize_(m, bits, group=None):
    """Symmetric weight-only quantisation of every Linear layer (per output row, or per group of `group` inputs).
    Weights are rounded to integers in [-(2^(b-1)-1), 2^(b-1)-1] and immediately de-quantised: this SIMULATES the
    accuracy effect. Real kernels keep the integers in memory (the speed and memory win) and dequantise on the fly."""
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, torch.nn.Linear) and mod is not m.lm_head:
                w = mod.weight
                wg = w.view(w.shape[0], -1, group) if group else w.view(w.shape[0], 1, -1)
                scale = wg.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
                mod.weight.copy_((wg / scale).round().clamp(-qmax, qmax).mul(scale).view_as(w))
    return m


section("Quantisation: fewer bits per weight (trained char GPT; weights of all Linear layers)")
base = val_loss(model)
n_lin = sum(mod.weight.numel() for mod in model.modules() if isinstance(mod, torch.nn.Linear) and mod is not model.lm_head)
print(f"{'format':28s} {'bytes for Linear weights':>24s} {'val loss':>9s}")
print(f"{'fp32 (as trained)':28s} {4 * n_lin:24,d} {base:9.4f}")
print(f"{'bf16':28s} {2 * n_lin:24,d} {val_loss(copy.deepcopy(model).to(torch.bfloat16).float()):9.4f}")
for bits, group, name in [(8, None, "int8, per-row scale"), (4, None, "int4, per-row scale"), (4, 32, "int4, groups of 32"),
                          (3, 32, "int3, groups of 32"), (2, 32, "int2, groups of 32")]:
    size = n_lin * bits / 8 + (n_lin / (group or 128)) * 2      # + one fp16 scale per group (per-row: ~1 per 128 here)
    print(f"{name:28s} {int(size):24,d} {val_loss(quantize_(copy.deepcopy(model), bits, group)):9.4f}")
print("-> 8-bit is essentially free; 4-bit costs a little; separate scales per group of 32 weights claw some of it")
print("   back (GGUF 'Q4' formats work this way; GPTQ/AWQ also choose the rounding to minimise output error).")
print("   Below 4 bits quality falls quickly.")


section("Speculative decoding: a small draft model proposes, the big model verifies")
print("training a 2-layer draft model on the same data (≈ 1 min)...")
torch.manual_seed(0)
draft = GPT(GPTConfig(vocab_size=len(vocab), block_size=128, n_layer=2, n_head=2, n_embd=64))
train(draft, train_data, val_data, max_steps=800, batch_size=32, lr=2e-3, warmup=50, eval_every=800, log=lambda *_: None)
print(f"target: {model.num_params():,} params, val {base:.3f} | draft: {draft.num_params():,} params, val {val_loss(draft):.3f}")


@torch.no_grad()
def greedy(m, idx, n):
    calls = 0
    for _ in range(n):
        idx = torch.cat([idx, m(idx[:, -128:])[0][:, -1].argmax(-1, keepdim=True)], 1)
        calls += 1
    return idx, calls


@torch.no_grad()
def speculative_greedy(target, draft, idx, n, k=4):
    """Draft proposes k tokens greedily; ONE target forward pass scores all of them; keep the longest prefix the
    target agrees with, plus the target's own next token. Output is identical to greedy decoding with the target."""
    calls, accepted, proposed, start = 0, 0, 0, idx.shape[1]
    while idx.shape[1] - start < n:
        prop = idx
        for _ in range(k):
            prop = torch.cat([prop, draft(prop[:, -128:])[0][:, -1].argmax(-1, keepdim=True)], 1)
        logits, _ = target(prop[:, -128:], prop[:, -128:])        # targets arg -> logits at every position
        calls += 1
        tgt = logits[0, -k - 1:].argmax(-1)                         # target's choice after each prefix
        drafted = prop[0, -k:]
        n_ok = int((tgt[:k] == drafted).long().cumprod(0).sum())
        accepted += n_ok
        proposed += k
        idx = torch.cat([idx, drafted[None, :n_ok], tgt[None, n_ok:n_ok + 1]], 1)
    return idx[:, :start + n], calls, accepted / proposed


prompt = torch.tensor([vocab.encode("ROMEO:\n")])
N_NEW = 120          # prompt + new tokens stay within block_size (128): no sliding window, so outputs must match exactly
t0 = time.perf_counter(); ref, ref_calls = greedy(model, prompt, N_NEW); t_ref = time.perf_counter() - t0
for k in [2, 4, 8]:
    t0 = time.perf_counter(); spec, calls, acc = speculative_greedy(model, draft, prompt, N_NEW, k); t_spec = time.perf_counter() - t0
    print(f"k={k}: identical output {torch.equal(spec, ref)} | target forward passes {calls:3d} vs {ref_calls} | "
          f"draft acceptance {acc:.2f} | wall time {t_spec:.2f} s vs {t_ref:.2f} s")
print("-> fewer passes through the big model for the same output. It pays off when the target is much more")
print("   expensive than the draft (and memory-bound, so verifying k tokens costs about as much as 1).")
print("   Here the target is only ~8x the draft and runs on a CPU, so it is SLOWER in wall-clock time.")
