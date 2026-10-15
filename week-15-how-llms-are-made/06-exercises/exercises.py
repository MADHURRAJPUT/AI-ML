"""Week 15 exercises — fill in each function, then run `python exercises.py`."""
import math
import sys

import torch
import torch.nn.functional as F


def chinchilla_optimal(compute_flops):
    """Return (N params, D tokens) minimising loss for C = 6·N·D with D = 20·N."""
    raise NotImplementedError


def training_days(n_params, n_tokens, n_gpus, peak_flops_per_gpu, mfu):
    """Wall-clock days to train with C = 6·N·D at a model-FLOPs utilisation of mfu (e.g. 0.4)."""
    raise NotImplementedError


def sft_example(prompt_ids, answer_ids):
    """(x, y) for SFT: x = (prompt + answer)[:-1], y = (prompt + answer)[1:], with every target that is part of
    the prompt replaced by -100 (so only answer tokens are trained)."""
    raise NotImplementedError


def bradley_terry_loss(r_chosen, r_rejected):
    """Reward-model loss: mean of -log σ(r_chosen - r_rejected)."""
    raise NotImplementedError


def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta):
    """All arguments: tensors of summed response log-probs, one per pair. Return the mean DPO loss."""
    raise NotImplementedError


def grpo_advantages(rewards, eps=1e-4):
    """rewards: (P prompts, G samples). Normalise within each group: (r - mean) / (std + eps).
    Use the unbiased std (torch default)."""
    raise NotImplementedError


def pass_at_k(n, c, k):
    """Unbiased pass@k (Chen et al. 2021): probability that at least one of k samples, drawn without replacement
    from n generated samples of which c are correct, is correct. = 1 - C(n-c, k) / C(n, k)."""
    raise NotImplementedError


def rmsnorm(x, weight, eps=1e-6):
    """x / sqrt(mean(x², last dim) + eps) * weight."""
    raise NotImplementedError


def moe_balance_loss(router_probs, chosen, n_experts):
    """Switch-Transformer auxiliary loss. router_probs: (N tokens, E) softmax outputs; chosen: (N, k) expert ids.
    f_i = fraction of the N·k routing slots that went to expert i; p_i = mean router probability of expert i.
    loss = E · Σ_i f_i · p_i."""
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
        N, D = mod.chinchilla_optimal(5.88e23)
        assert abs(N / 70e9 - 1) < 0.01 and abs(D / 1.4e12 - 1) < 0.01, (N, D)

    def t2():
        d = mod.training_days(175e9, 300e9, 1000, 312e12, 0.4)
        assert abs(d - 6 * 175e9 * 300e9 / (1000 * 312e12 * 0.4) / 86400) < 1e-6 and 29 < d < 30

    def t3():
        x, y = mod.sft_example([1, 2, 3], [7, 8, 9])
        assert list(x) == [1, 2, 3, 7, 8] and list(y) == [-100, -100, 7, 8, 9], (x, y)

    def t4():
        l = mod.bradley_terry_loss(torch.tensor([2.0, 0.0]), torch.tensor([0.0, 0.0]))
        assert abs(l.item() - (-(F.logsigmoid(torch.tensor(2.0)) + F.logsigmoid(torch.tensor(0.0))) / 2).item()) < 1e-6

    def t5():
        z = torch.zeros(4)
        assert abs(mod.dpo_loss(z, z, z, z, 0.1).item() - math.log(2)) < 1e-6
        l = mod.dpo_loss(torch.tensor([-5.0]), torch.tensor([-20.0]), torch.tensor([-10.0]), torch.tensor([-10.0]), 0.1)
        assert abs(l.item() - (-F.logsigmoid(torch.tensor(0.1 * (5 + 10)))).item()) < 1e-6

    def t6():
        a = mod.grpo_advantages(torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]))
        assert torch.allclose(a[1], torch.zeros(4)), "a group with equal rewards has zero advantage"
        assert torch.allclose(a[0], torch.tensor([1.0, -1.0, -1.0, 1.0]) * 0.5 / (0.57735 + 1e-4), atol=1e-3)

    def t7():
        assert mod.pass_at_k(10, 0, 5) == 0 and mod.pass_at_k(10, 10, 1) == 1
        assert abs(mod.pass_at_k(10, 3, 1) - 0.3) < 1e-12
        assert abs(mod.pass_at_k(10, 3, 5) - (1 - math.comb(7, 5) / math.comb(10, 5))) < 1e-12
        assert mod.pass_at_k(10, 8, 5) == 1.0, "when n - c < k, some correct sample is always drawn"

    def t8():
        x, w = torch.randn(3, 8) * 4, torch.rand(8)
        ref = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
        assert torch.allclose(mod.rmsnorm(x, w), ref, atol=1e-6)

    def t9():
        E = 4
        uniform = mod.moe_balance_loss(torch.full((8, E), 0.25), torch.tensor([[0, 1], [2, 3]] * 4), E)
        assert abs(uniform.item() - 1.0) < 1e-6, "balanced routing gives loss 1"
        p = torch.tensor([[0.7, 0.1, 0.1, 0.1]] * 8)
        skewed = mod.moe_balance_loss(p, torch.tensor([[0, 1]] * 8), E)
        assert abs(skewed.item() - E * (0.5 * 0.7 + 0.5 * 0.1)) < 1e-6 and skewed > uniform

    names = ["chinchilla_optimal", "training_days", "sft_example", "bradley_terry_loss", "dpo_loss",
             "grpo_advantages", "pass_at_k", "rmsnorm", "moe_balance_loss"]
    for i, (name, t) in enumerate(zip(names, [t1, t2, t3, t4, t5, t6, t7, t8, t9]), 1):
        run(f"{i}. {name:19s}", t)
    for name, status in results:
        print(f"{name}  {status}")
    passed = sum(s == "PASS" for _, s in results)
    print(f"\n{passed}/{len(results)} passing")


if __name__ == "__main__":
    _check(sys.modules[__name__])
