"""Week 13 exercises — fill in each function, then run `python exercises.py`."""
import math
import sys

import torch
import torch.nn.functional as F


def attention(q, k, v, causal=False):
    """Scaled dot-product attention. q: (..., Tq, d), k, v: (..., Tk, d). Do NOT call F.scaled_dot_product_attention.
    If causal, query i may only attend to keys 0..i (assume Tq == Tk)."""
    raise NotImplementedError


def split_heads(x, H):
    """(B, T, C) -> (B, H, T, C // H). Head h must hold channels h*d .. (h+1)*d - 1."""
    raise NotImplementedError


def merge_heads(x):
    """Inverse of split_heads: (B, H, T, d) -> (B, T, H*d)."""
    raise NotImplementedError


def mha_forward(x, w_qkv, b_qkv, w_o, b_o, H):
    """Causal multi-head self-attention with GPT-2 conventions:
    qkv = x @ w_qkv.T + b_qkv (w_qkv: (3C, C), rows ordered q, k, v); out = merged @ w_o.T + b_o.
    Use your attention / split_heads / merge_heads."""
    raise NotImplementedError


def sinusoidal_pe(T, d):
    """(T, d) tensor: PE[t, 2i] = sin(t / 10000^(2i/d)), PE[t, 2i+1] = cos(t / 10000^(2i/d))."""
    raise NotImplementedError


def apply_rope(x, base=10000.0):
    """Rotary embedding for x of shape (T, d): rotate each pair (x[t, 2i], x[t, 2i+1]) by angle t * base^(-2i/d)."""
    raise NotImplementedError


def block_params(C):
    """Exact parameter count of one GPT-2 block (2 LayerNorms with weight+bias, c_attn C->3C, c_proj C->C,
    c_fc C->4C, mlp c_proj 4C->C, all Linear layers with bias)."""
    raise NotImplementedError


def gpt2_params(V, T, C, L):
    """Exact parameter count of GPT-2: wte (V, C), wpe (T, C), L blocks, final LayerNorm, lm_head tied to wte."""
    raise NotImplementedError


def kv_cache_bytes(n_layers, n_kv_heads, head_dim, n_tokens, batch=1, bytes_per_value=2):
    """Memory of the KV cache: K and V for every layer, KV head, token and sequence in the batch."""
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

    torch.manual_seed(0)

    def t1():
        q, k, v = torch.randn(3, 2, 4, 7, 8).unbind(0)
        for causal in [False, True]:
            ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            assert torch.allclose(mod.attention(q, k, v, causal), ref, atol=1e-5), f"causal={causal}"
        q2 = torch.randn(1, 3, 8)                          # cross-attention shapes: Tq != Tk
        assert mod.attention(q2, k[0, :1], v[0, :1]).shape == (1, 3, 8)

    def t2():
        x = torch.randn(2, 5, 12)
        s = mod.split_heads(x, 3)
        assert s.shape == (2, 3, 5, 4)
        assert torch.equal(s[:, 1], x[..., 4:8]), "head 1 should hold channels 4..7"
        assert torch.equal(mod.merge_heads(s), x)

    def t3():
        C, H, T = 16, 4, 6
        x = torch.randn(2, T, C)
        ref = torch.nn.MultiheadAttention(C, H, batch_first=True)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        out = mod.mha_forward(x, ref.in_proj_weight, ref.in_proj_bias, ref.out_proj.weight, ref.out_proj.bias, H)
        assert torch.allclose(out, ref(x, x, x, attn_mask=mask, need_weights=False)[0], atol=1e-5)

    def t4():
        pe = mod.sinusoidal_pe(50, 16)
        assert pe.shape == (50, 16)
        t = torch.arange(50.0)
        assert torch.allclose(pe[:, 0], torch.sin(t), atol=1e-5) and torch.allclose(pe[:, 1], torch.cos(t), atol=1e-5)
        assert abs(pe[3, 2].item() - math.sin(3 / 10000 ** (2 / 16))) < 1e-5

    def t5():
        T, d = 40, 16
        q, k = torch.randn(d), torch.randn(d)
        Q, K = mod.apply_rope(q.expand(T, d).clone()), mod.apply_rope(k.expand(T, d).clone())
        assert torch.allclose(Q[0], q, atol=1e-6), "position 0 must be unchanged"
        assert not torch.allclose(Q[1], q), "position 1 must be rotated"
        assert torch.allclose(Q.norm(dim=1), q.norm().expand(T), atol=1e-4), "rotation preserves norms"
        s = Q @ K.T
        assert torch.allclose(s.diagonal(-5), s[5, 0].expand(T - 5), atol=1e-4), "score must depend on offset only"
        assert abs(Q[1, 0].item() - (q[0] * math.cos(1) - q[1] * math.sin(1)).item()) < 1e-5, "pair (0, 1), angle t"

    def t6():
        assert mod.block_params(768) == 7_087_872
        assert mod.block_params(64) == 49_984

    def t7():
        assert mod.gpt2_params(50257, 1024, 768, 12) == 124_439_808
        assert mod.gpt2_params(65, 256, 384, 6) == 10_770_816

    def t8():
        assert mod.kv_cache_bytes(32, 8, 128, 131072) == 17_179_869_184     # 16 GiB (≈ 17.2 GB)
        assert mod.kv_cache_bytes(12, 12, 64, 1024, batch=4, bytes_per_value=4) == 301_989_888

    names = ["attention", "split/merge heads", "mha_forward", "sinusoidal_pe", "apply_rope", "block_params",
             "gpt2_params", "kv_cache_bytes"]
    for i, (name, t) in enumerate(zip(names, [t1, t2, t3, t4, t5, t6, t7, t8]), 1):
        run(f"{i}. {name:18s}", t)
    for name, status in results:
        print(f"{name}  {status}")
    passed = sum(s == "PASS" for _, s in results)
    print(f"\n{passed}/{len(results)} passing")


if __name__ == "__main__":
    _check(sys.modules[__name__])
