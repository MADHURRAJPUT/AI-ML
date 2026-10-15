"""Week 15 reference solutions. Try exercises.py first!"""
import math
import sys

import torch
import torch.nn.functional as F

from exercises import _check


def chinchilla_optimal(compute_flops):
    N = math.sqrt(compute_flops / (6 * 20))       # C = 6 · N · 20N = 120 N²
    return N, 20 * N


def training_days(n_params, n_tokens, n_gpus, peak_flops_per_gpu, mfu):
    return 6 * n_params * n_tokens / (n_gpus * peak_flops_per_gpu * mfu) / 86400


def sft_example(prompt_ids, answer_ids):
    ids = list(prompt_ids) + list(answer_ids)
    x, y = ids[:-1], ids[1:]
    return x, [-100] * (len(prompt_ids) - 1) + y[len(prompt_ids) - 1:]


def bradley_terry_loss(r_chosen, r_rejected):
    return -F.logsigmoid(r_chosen - r_rejected).mean()


def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta):
    return -F.logsigmoid(beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))).mean()


def grpo_advantages(rewards, eps=1e-4):
    return (rewards - rewards.mean(1, keepdim=True)) / (rewards.std(1, keepdim=True) + eps)


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1 - math.comb(n - c, k) / math.comb(n, k)


def rmsnorm(x, weight, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def moe_balance_loss(router_probs, chosen, n_experts):
    f = F.one_hot(chosen, n_experts).float().sum((0, 1)) / chosen.numel()
    p = router_probs.mean(0)
    return n_experts * (f * p).sum()


if __name__ == "__main__":
    _check(sys.modules[__name__])
