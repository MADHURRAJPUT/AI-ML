# 12.5 Week 12 exercises

```bash
python exercises.py      # checks your implementations
python solutions.py      # reference solutions
```

| # | Function | Skill |
|---|---|---|
| 1 | `get_stats(ids)` | adjacent pair counts |
| 2 | `merge(ids, pair, idx)` | the replace step |
| 3 | `train_bpe(text, vocab_size)` | the whole training loop |
| 4 | `encode` / `decode` | merge order; byte-level round-trips |
| 5 | `chat_token_count(messages, tok)` | chat templates + the special-token injection guard |
| 6 | `estimate_monthly_cost(...)` | pricing a feature |

## The real exercise: BPE from memory

Close `02-bpe-from-scratch/bpe_basic.py` and write `train`, `encode`, `decode`
from a blank file. Train on Shakespeare with vocab 512 and check that your
encoder round-trips the whole text. Then add regex pre-tokenisation and
special tokens (compare with `common/bpe.py`). Budget: 2 hours.

## Interview drill

- "What is BPE and why do LLMs use subword tokenisation?"
- "Why are LLMs bad at counting letters or at arithmetic?"
- "Why does the same prompt cost more in Japanese than in English?"
- "How would you estimate the monthly API cost of a new LLM feature?"
