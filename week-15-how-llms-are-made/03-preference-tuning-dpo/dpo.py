"""Week 15.3 — preference tuning with DPO: teach the SFT model to refuse one kind of request, from preference
pairs alone, while staying close to the reference model. Needs runs/sft_gpt.pt (15.2). ~3 min."""
import copy
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.chat_tasks import END, TASKS, accuracy, collate, prompt, word_lists  # noqa: E402
from common.gpt import GPT  # noqa: E402
from common.text_data import CharVocab, load_text  # noqa: E402

CKPT = ROOT / "runs" / "sft_gpt.pt"
if not CKPT.exists():
    sys.exit(f"{CKPT} not found — run 02-supervised-fine-tuning/sft.py first.")


def section(title):
    print(f"\n=== {title} ===")


policy, ckpt = GPT.load(CKPT)
vocab = CharVocab("".join(ckpt["chars"]))
reference = copy.deepcopy(policy).eval()            # frozen copy of the SFT model: the "anchor"
for p in reference.parameters():
    p.requires_grad_(False)
train_words, test_words = word_lists(load_text("shakespeare"))

CHOSEN = lambda w: f"I will not curse {w}.{END}"        # noqa: E731  — the behaviour we prefer
REJECTED = lambda w: f"a plague on {w}!{END}"            # noqa: E731  — the behaviour we don't
REFUSAL = lambda task, w: f"I will not {task} {w}.{END}"  # noqa: E731  — an over-refusal on a harmless request


def curse_pair(w):
    return "curse", w, CHOSEN(w), REJECTED(w)


def helpful_pair(rng, w):
    """A harmless request: chosen = the correct answer, rejected = refusing it."""
    task = rng.choice(list(TASKS))
    return task, w, TASKS[task](w) + END, REFUSAL(task, w)


def encode_side(task, word, response):
    p, a = vocab.encode(prompt(task, word)), vocab.encode(response)
    ids = p + a
    return ids[:-1], [-100] * (len(p) - 1) + ids[len(p):]


def seq_logprob(model, X, Y):
    """Sum of log-probabilities of the response tokens (prompt positions are -100), and the token count."""
    logits, _ = model(X, Y)
    mask = Y != -100
    tok = F.log_softmax(logits, -1).gather(-1, Y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (tok * mask).sum(-1), mask.sum(-1)


def pair_logprobs(model, pairs):
    pad = vocab.stoi["\n"]
    Xc, Yc = collate([encode_side(t, w, c) for t, w, c, _ in pairs], pad)
    Xr, Yr = collate([encode_side(t, w, r) for t, w, _, r in pairs], pad)
    (lc, nc), (lr, _) = seq_logprob(model, Xc, Yc), seq_logprob(model, Xr, Yr)
    return lc, lr, nc


@torch.no_grad()
def respond(model, task, w):
    idx = torch.tensor([vocab.encode(prompt(task, w))])
    return vocab.decode(model.generate(idx, 30, temperature=0, eos_id=vocab.stoi[END])[0, idx.shape[1]:].tolist())


def report(model, label):
    outs = [respond(model, "curse", w) for w in test_words[:100]]
    refuse = sum(o.startswith("I will not curse") for o in outs) / 100
    curse = sum(o.startswith("a plague") for o in outs) / 100
    over = sum(respond(model, t, w).startswith("I will not") for t in TASKS for w in test_words[:30]) / 90
    acc = accuracy(model, vocab, test_words[:150])
    x = torch.tensor([vocab.encode(prompt("reverse", w) + w[::-1] + END) for w in test_words[:60] if len(w) == 5])
    with torch.no_grad():
        lp, lq = F.log_softmax(model(x, x)[0], -1), F.log_softmax(reference(x, x)[0], -1)
    kl = (lp.exp() * (lp - lq)).sum(-1).mean().item()
    row = (label, refuse, curse, over, acc["upper"], acc["reverse"], acc["spell"], kl)
    print(f"  'curse' prompts: refuses {refuse:.0%}, curses {curse:.0%} | harmless prompts refused: {over:.0%}")
    print(f"  skills: upper {acc['upper']:.2f}, reverse {acc['reverse']:.2f}, spell {acc['spell']:.2f} | "
          f"KL to reference on ordinary chats {kl:.3f} nats/token")
    print(f"  e.g. curse {test_words[0]!r} -> {outs[0]!r} | reverse {test_words[1]!r} -> "
          f"{respond(model, 'reverse', test_words[1])!r}")
    return row


def run_dpo(lr, steps, nll_weight, mixed, beta=0.1, seed=0):
    """DPO: loss = -log σ(β [(log π(c) - log π_ref(c)) - (log π(r) - log π_ref(r))])
    + nll_weight * (per-token negative log-likelihood of the chosen response)."""
    torch.manual_seed(seed)
    model = copy.deepcopy(reference).requires_grad_(True).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    for step in range(steps + 1):
        words = [rng.choice(train_words) for _ in range(32)]
        pairs = [curse_pair(w) if (not mixed or i % 2 == 0) else helpful_pair(rng, w) for i, w in enumerate(words)]
        pc, pr, nc = pair_logprobs(model, pairs)
        with torch.no_grad():
            rc, rr, _ = pair_logprobs(reference, pairs)
        margin = beta * ((pc - rc) - (pr - rr))            # implicit reward: chosen minus rejected
        loss = -F.logsigmoid(margin).mean()
        if nll_weight:
            loss = loss + nll_weight * (-pc / nc).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"  step {step:3d} | loss {loss.item():.4f} | reward margin {margin.mean().item():6.2f} | "
                  f"log P(chosen) {pc.mean().item():7.1f} | log P(rejected) {pr.mean().item():7.1f}")
    return model.eval()


section("Preference data: (prompt, chosen, rejected)")
print(repr(prompt("curse", "romeo") + CHOSEN("romeo")), "  <- chosen")
print(repr(prompt("curse", "romeo") + REJECTED("romeo")), "  <- rejected")
print("The SFT model has never seen a 'curse' request, and we never give it an SFT target: only pairs.")

section("Before DPO (the SFT model = the reference)")
rows = [report(reference, "SFT model (reference)")]

experiments = [
    ("A. plain DPO, 'curse' pairs only", dict(lr=1e-4, steps=150, nll_weight=0.0, mixed=False)),
    ("B. DPO + NLL on chosen, 'curse' pairs only", dict(lr=1e-4, steps=150, nll_weight=1.0, mixed=False)),
    ("C. DPO + NLL, 50% 'curse' + 50% harmless pairs", dict(lr=1e-4, steps=150, nll_weight=1.0, mixed=True)),
]
for label, kw in experiments:
    section(label)
    rows.append(report(run_dpo(**kw), label))

section("Summary (100 held-out 'curse' prompts; 90 held-out harmless prompts; 150 words per skill)")
print(f"{'':48s} {'refuse':>6s} {'curse':>6s} {'over-refuse':>11s} {'upper':>6s} {'reverse':>7s} {'spell':>6s} {'KL':>6s}")
for label, r, c, o, u, rv, sp, kl in rows:
    print(f"{label:48s} {r:6.0%} {c:6.0%} {o:11.0%} {u:6.2f} {rv:7.2f} {sp:6.2f} {kl:6.3f}")
