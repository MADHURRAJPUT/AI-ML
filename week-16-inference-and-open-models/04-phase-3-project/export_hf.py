"""Phase 3 project — export the names GPT as a Hugging Face model folder (+ model card) and reload it.

    python export_hf.py       # -> runs/names_gpt_hf/ ; loads it back with AutoModelForCausalLM and generates
"""
import sys
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from common.gpt import GPT, to_hf  # noqa: E402
from data import SEP  # noqa: E402

warnings.filterwarnings("ignore")
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast  # noqa: E402
from transformers import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

model, ckpt = GPT.load(ROOT / "runs" / "names_gpt.pt")
chars = ckpt["chars"]
out = ROOT / "runs" / "names_gpt_hf"

hf = to_hf(model.eval(), eos_id=chars.index(SEP))
tok = Tokenizer(models.WordLevel({c: i for i, c in enumerate(chars)}, unk_token=None))
tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
tok.decoder = decoders.Fuse()
hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token=SEP, eos_token=SEP, pad_token=SEP)
hf.save_pretrained(out)
hf_tok.save_pretrained(out)

hist = ckpt["history"][-1]
(out / "README.md").write_text(f"""---
license: mit
tags: [gpt2, character-level, names, educational]
---
# names-gpt (Phase 3 project)

A {model.num_params():,}-parameter character-level GPT-2-architecture model that generates English first names.

## Training
- **Data:** 29 494 unique names from Karpathy's makemore `names.txt`; 90 % train / 10 % held out, split by name.
- **Architecture:** {model.config.n_layer} layers, {model.config.n_head} heads, d_model {model.config.n_embd}, context {model.config.block_size}, vocabulary {len(chars)} (a-z and newline as end-of-name).
- **Training:** {hist['step']} steps, batch 64, AdamW (lr 2e-3, warm-up + cosine), dropout {model.config.dropout}. CPU, about 6 minutes.

## Evaluation (held-out names)
- Loss: {hist['val']:.3f} nats/char ({hist['val'] / 0.6931:.2f} bits/char) vs 3.54 bits/char for a count-based bigram model.
- Of 1 000 samples at temperature 1.0: about 75 % are new names, 24 % copy a training name, and 1.6 % are real names from the held-out set.

## Use
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("names_gpt_hf"); model = AutoModelForCausalLM.from_pretrained("names_gpt_hf")
ids = tok("\\n", return_tensors="pt").input_ids
out = model.generate(ids, do_sample=True, max_new_tokens=20, eos_token_id=tok.eos_token_id)
```

## Limitations
- English-centric, US-style names only; the frequencies mirror the source list, not any real population.
- About a quarter of samples copy training names: it has partly memorised its data.
- Not suitable for anything that requires culturally appropriate or verified names.
""")
print("wrote", sorted(p.name for p in out.iterdir()))

m = AutoModelForCausalLM.from_pretrained(out).eval()
t = AutoTokenizer.from_pretrained(out)
ids = t(SEP, return_tensors="pt").input_ids
torch.manual_seed(0)
with torch.no_grad():
    outs = m.generate(ids, do_sample=True, max_new_tokens=20, num_return_sequences=10,
                      eos_token_id=t.eos_token_id, pad_token_id=t.pad_token_id)
print("reloaded with AutoModelForCausalLM; 10 samples:",
      [t.decode(o[1:]).split(SEP)[0] for o in outs])
