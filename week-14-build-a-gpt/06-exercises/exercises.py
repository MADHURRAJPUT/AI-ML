"""Week 14 exercises — fill in each function, then run `python exercises.py`."""
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.gpt import GPT, GPTConfig  # noqa: E402


def get_batch(data, block_size, batch_size, generator):
    """Random windows: x (B, T) and y = x shifted left by one. Start indices from
    torch.randint(len(data) - block_size, (batch_size,), generator=generator)."""
    raise NotImplementedError


def lr_at(step, max_steps, lr, warmup, min_lr):
    """Linear warm-up from lr/warmup at step 0 to lr at step warmup-1; then cosine decay from lr (at step
    `warmup`) to min_lr (at step max_steps)."""
    raise NotImplementedError


def decay_param_names(model):
    """Names of parameters that SHOULD get weight decay (dim >= 2), sorted. Tied weights appear once."""
    raise NotImplementedError


def temperature_probs(logits, T):
    """Softmax of logits / T. T == 0 -> one-hot on the argmax."""
    raise NotImplementedError


def top_k_filter(logits, k):
    """Set every logit outside the k largest to -inf (1-D logits)."""
    raise NotImplementedError


def top_p_filter(logits, p):
    """Nucleus: keep the smallest set of highest-probability tokens whose total probability is >= p;
    set the rest to -inf (1-D logits)."""
    raise NotImplementedError


def bits_per_char(loss_nats_per_token, chars_per_token):
    raise NotImplementedError


def greedy_with_cache(model, idx, n_new):
    """Greedy generation using model(..., past_kv=..., use_cache=True), which returns (logits, loss, present).
    First call: the whole prompt. Then feed ONLY the newest token each step. Return the full (1, T+n_new) ids."""
    raise NotImplementedError


# ---------------------------------------------------------------- checks ----
def _check(mod):
    results = []

    def run(name, fn):
        try:
            fn(); results.append((name, "PASS"))
        except NotImplementedError:
            results.append((name, "todo"))
        except Exception as e:  # noqa: BLE001
            results.append((name, f"FAIL: {type(e).__name__}: {e}"))

    def t1():
        data = torch.arange(100)
        x, y = mod.get_batch(data, 8, 4, torch.Generator().manual_seed(0))
        assert x.shape == y.shape == (4, 8)
        assert torch.equal(y, x + 1), "y must be x shifted by one"
        assert x.max() + 1 < 100

    def t2():
        f = lambda s: mod.lr_at(s, 1000, 1e-3, 100, 1e-4)  # noqa: E731
        assert abs(f(0) - 1e-5) < 1e-12 and abs(f(99) - 1e-3) < 1e-12
        assert abs(f(100) - 1e-3) < 1e-12 and abs(f(1000) - 1e-4) < 1e-12
        assert abs(f(550) - (1e-4 + 0.5 * 9e-4)) < 1e-12, "halfway through the decay: midpoint"

    def t3():
        m = GPT(GPTConfig(vocab_size=11, block_size=8, n_layer=1, n_head=2, n_embd=8))
        names = mod.decay_param_names(m)
        assert names == sorted(["transformer.wte.weight", "transformer.wpe.weight", "transformer.h.0.attn.c_attn.weight",
                                "transformer.h.0.attn.c_proj.weight", "transformer.h.0.mlp.c_fc.weight",
                                "transformer.h.0.mlp.c_proj.weight"]), names

    def t4():
        logits = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(mod.temperature_probs(logits, 1.0), F.softmax(logits, -1))
        assert torch.equal(mod.temperature_probs(logits, 0), torch.tensor([0.0, 0.0, 1.0]))
        assert mod.temperature_probs(logits, 0.5)[2] > mod.temperature_probs(logits, 2.0)[2]

    def t5():
        out = mod.top_k_filter(torch.tensor([0.1, 3.0, 2.0, -1.0]), 2)
        assert torch.equal(torch.isinf(out), torch.tensor([True, False, False, True])) and out[1] == 3.0

    def t6():
        logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
        keep = lambda p: (~torch.isinf(mod.top_p_filter(logits, p))).tolist()  # noqa: E731
        assert keep(0.5) == [True, False, False, False]
        assert keep(0.6) == [True, True, False, False]
        assert keep(0.9) == [True, True, True, False]
        shuffled = mod.top_p_filter(logits[[2, 0, 3, 1]], 0.6)
        assert (~torch.isinf(shuffled)).tolist() == [False, True, False, True], "must work on unsorted logits"

    def t7():
        assert abs(mod.bits_per_char(math.log(2) * 3, 1.5) - 2.0) < 1e-9

    def t8():
        torch.manual_seed(0)
        m = GPT(GPTConfig(vocab_size=30, block_size=40, n_layer=2, n_head=2, n_embd=16)).eval()
        idx = torch.tensor([[1, 2, 3]])
        assert torch.equal(mod.greedy_with_cache(m, idx, 20), m.generate(idx, 20, temperature=0, use_cache=False))

    names = ["get_batch", "lr_at", "decay_param_names", "temperature_probs", "top_k_filter", "top_p_filter",
             "bits_per_char", "greedy_with_cache"]
    for i, (name, t) in enumerate(zip(names, [t1, t2, t3, t4, t5, t6, t7, t8]), 1):
        run(f"{i}. {name:18s}", t)
    for name, status in results:
        print(f"{name}  {status}")
    passed = sum(s == "PASS" for _, s in results)
    print(f"\n{passed}/{len(results)} passing")


if __name__ == "__main__":
    _check(sys.modules[__name__])
