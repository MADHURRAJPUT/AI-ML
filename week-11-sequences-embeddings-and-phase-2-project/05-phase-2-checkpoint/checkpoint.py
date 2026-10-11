"""Phase 2 checkpoint — diagnose six mystery loss curves.

    python checkpoint.py            # shows the curves; write down cause + fix for each
    python checkpoint.py --answers  # reveals what was broken

Reuses week 9.6's experiment runner, with the labels hidden and the order shuffled.
"""
import argparse
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "week-09-training-dynamics" / "06-training-debugging-log"))
from break_training import EXPERIMENTS, run  # noqa: E402

BARS = "▁▂▃▄▅▆▇█"
PICK = ["LR far too high (Adam lr=0.1)", "LR far too low (Adam lr=1e-6)", "SGD lr=1.0 (diverges)",
        "labels shuffled (data bug)", "only 200 training examples", "forgot optimizer.zero_grad()"]


def sparkline(values, width=60, lo=None, hi=None):
    chunks = [values[int(i * len(values) / width):int((i + 1) * len(values) / width)] or [values[-1]]
              for i in range(width)]
    pts = [sum(c) / len(c) for c in chunks]
    pts = [p if math.isfinite(p) else float("inf") for p in pts]
    finite = [p for p in pts if math.isfinite(p)] or [0.0]
    lo = min(finite) if lo is None else lo
    hi = max(finite) if hi is None else hi
    out = ""
    for p in pts:
        if not math.isfinite(p):
            out += "✖"
        else:
            out += BARS[min(len(BARS) - 1, int((math.log(p + 1e-9) - math.log(lo + 1e-9)) /
                                               (math.log(hi + 1e-9) - math.log(lo + 1e-9) + 1e-12) * (len(BARS) - 1)))]
    return out, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true")
    args = ap.parse_args()
    chosen = [e for e in EXPERIMENTS if e[0] in PICK]
    random.Random(7).shuffle(chosen)
    print("Train loss (log scale, ✖ = NaN) over 600 steps, and validation loss at the end.")
    print("For each curve write: (1) what you think is broken, (2) how you'd confirm it, (3) the fix.\n")
    for k, (name, kw, note) in enumerate(chosen, 1):
        r = run(name, **kw)
        line, lo, hi = sparkline(r["train"])
        val = r["val"][-1][1]
        first = sum(r["train"][:20]) / 20
        print(f"curve {chr(64 + k)}: {line}")
        print(f"         train loss {first:.3g} -> {sum(r['train'][-50:]) / 50:.3g} | final val loss "
              f"{val:.3g} | val acc {r['acc']:.3f}  (ln 10 = 2.30)\n")
        if args.answers:
            print(f"         ANSWER: {name}\n         {note}\n")
    if not args.answers:
        print("Run again with --answers to check yourself. Aim for 6/6 with a correct fix for each.")


if __name__ == "__main__":
    main()
