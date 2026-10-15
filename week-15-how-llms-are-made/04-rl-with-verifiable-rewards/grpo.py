"""Week 15.4 — reinforcement learning with verifiable rewards (GRPO), on the SFT model from 15.2.

The model samples several answers per prompt; a program checks each one (reward 1 if exactly right, else 0);
answers that beat their group's average are made more likely. No labels are shown, only rewards. ~6 min.
"""
import copy
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.chat_tasks import END, TASKS, accuracy, prompt, word_lists  # noqa: E402
from common.gpt import GPT  # noqa: E402
from common.text_data import load_text  # noqa: E402
from common.text_data import CharVocab  # noqa: E402

CKPT = ROOT / "runs" / "sft_gpt.pt"
if not CKPT.exists():
    sys.exit(f"{CKPT} not found — run 02-supervised-fine-tuning/sft.py first.")


def section(title):
    print(f"\n=== {title} ===")


policy, ckpt = GPT.load(CKPT)
vocab = CharVocab("".join(ckpt["chars"]))
EOS = vocab.stoi[END]
reference = copy.deepcopy(policy).eval().requires_grad_(False)
train_words, test_words = word_lists(load_text("shakespeare"))


def reward(task, word, completion):
    """The verifier: a program, not a learned model. 1.0 if the answer is exactly right."""
    return float(completion == TASKS[task](word))


@torch.no_grad()
def sample_group(model, task, word, G, generator, max_len=20):
    """G sampled completions for one prompt. Returns token tensors (G, P+L), the prompt length, decoded answers."""
    idx = torch.tensor([vocab.encode(prompt(task, word))]).repeat(G, 1)
    out = model.generate(idx, max_len, temperature=1.0, generator=generator, eos_id=EOS)
    texts = [vocab.decode(row[idx.shape[1]:].tolist()).split(END)[0] for row in out]
    return out, idx.shape[1], texts


def completion_mask(seqs, P):
    """1 for completion tokens up to and including the first END; 0 for the prompt and anything after."""
    comp = seqs[:, P:]
    is_eos = (comp == EOS).int()
    after_first_eos = (is_eos.cumsum(1) - is_eos) > 0
    mask = torch.zeros_like(seqs, dtype=torch.bool)
    mask[:, P:] = ~after_first_eos
    return mask[:, 1:]                              # aligned with next-token targets


def token_logprobs(model, seqs):
    logits, _ = model(seqs[:, :-1], seqs[:, 1:])    # passing targets returns logits for every position
    return F.log_softmax(logits, -1).gather(-1, seqs[:, 1:, None]).squeeze(-1)


section("Before RL: the SFT model")
before = accuracy(policy, vocab, test_words)
print("greedy accuracy on 300 held-out words:", {k: round(v, 3) for k, v in before.items()})
g = torch.Generator().manual_seed(0)
_, _, texts = sample_group(policy, "reverse", "tempest", 8, g)
print("8 samples (T=1) for 'reverse tempest':", texts)
print("rewards:", [reward("reverse", "tempest", t) for t in texts])
print("-> some samples are right and some are wrong. RL pushes probability towards the right ones.")

section("GRPO: group-relative advantages, no value network")
print("for each prompt: sample G answers, reward each, advantage = (r - mean(group)) / (std(group) + eps)")
print("loss = -mean over tokens of [advantage x log π(token)]  +  β · KL(π || π_ref)")
print("groups where every answer gets the same reward (all right or all wrong) give zero advantage: no signal.")

G, PROMPTS, STEPS, BETA = 8, 16, 120, 0.02
opt = torch.optim.AdamW(policy.parameters(), lr=2e-5, weight_decay=0.0)
rng, gen = random.Random(0), torch.Generator().manual_seed(0)
t0 = time.time()
for step in range(STEPS + 1):
    seqs_all, adv_all, masks_all, rewards_all, useful = [], [], [], [], 0
    for _ in range(PROMPTS):
        task, word = rng.choice(list(TASKS)), rng.choice(train_words)
        seqs, P, texts = sample_group(policy, task, word, G, gen)
        r = torch.tensor([reward(task, word, t) for t in texts])
        rewards_all.append(r.mean().item())
        if r.std() == 0:
            continue                                # no learning signal from this group
        useful += 1
        adv = (r - r.mean()) / (r.std() + 1e-4)
        seqs_all.append(seqs); adv_all.append(adv); masks_all.append(completion_mask(seqs, P))
    if seqs_all:
        policy.train()
        loss_total = 0.0
        for seqs, adv, mask in zip(seqs_all, adv_all, masks_all):
            logp = token_logprobs(policy, seqs)
            with torch.no_grad():
                ref_logp = token_logprobs(reference, seqs)
            ratio = ref_logp - logp                                  # k3 KL estimator (as in GRPO / DeepSeekMath)
            kl = ratio.exp() - ratio - 1
            per_token = -(adv[:, None] * logp) + BETA * kl
            loss = (per_token * mask).sum() / mask.sum() / len(seqs_all)
            loss.backward()
            loss_total += loss.item()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        policy.eval()
    if step % 20 == 0:
        print(f"step {step:3d} | mean reward of samples (T=1) {sum(rewards_all) / len(rewards_all):.3f} | "
              f"groups with signal {useful:2d}/{PROMPTS} | {time.time() - t0:4.0f} s")

section("After RL")
after = accuracy(policy, vocab, test_words)
print(f"{'task':8s} {'SFT':>6s} {'SFT + GRPO':>11s}")
for t in TASKS:
    print(f"{t:8s} {before[t]:6.3f} {after[t]:11.3f}")
g = torch.Generator().manual_seed(0)
_, _, texts = sample_group(policy, "reverse", "tempest", 8, g)
print("8 samples (T=1) for 'reverse tempest':", texts)
print("-> trained only on rewards from a checker, on training words; measured on held-out words.")

section("pass@k: what the SFT model could already do")
gen = torch.Generator().manual_seed(1)
for name, m in [("SFT", reference), ("SFT + GRPO", policy)]:
    p1 = pk = 0
    words = test_words[:100]
    for w in words:
        _, _, texts = sample_group(m, "reverse", w, 8, gen)
        rs = [reward("reverse", w, t) for t in texts]
        p1 += sum(rs) / 8
        pk += max(rs)
    print(f"{name:11s} reverse, 100 held-out words: pass@1 (T=1) {p1 / len(words):.3f} | pass@8 {pk / len(words):.3f}")
print("-> RL mostly turns 'right sometimes' (pass@8) into 'right first time' (pass@1). It sharpens what the")
print("   model can already sample; it rarely creates abilities that never appear in the samples.")
