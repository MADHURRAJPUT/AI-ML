"""Week 16.2 — the Hugging Face `transformers` workflow, on OUR models converted to HF format.

Everything here is the same API you use with downloaded open-weight models (see real_models.py); we use our own
checkpoints so the lesson runs offline and you can verify every number against code you wrote. ~30 s.
"""
import json
import sys
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.chat_tasks import END  # noqa: E402
from common.gpt import GPT, to_hf  # noqa: E402
from common.text_data import CharVocab  # noqa: E402

warnings.filterwarnings("ignore")
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast,  # noqa: E402
                          logging as hf_logging, pipeline)

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
for name in ["char_gpt.pt", "sft_gpt.pt"]:
    if not (ROOT / "runs" / name).exists():
        sys.exit(f"runs/{name} not found — run week 14.2 and week 15.2 first.")


def section(title):
    print(f"\n=== {title} ===")


def char_tokenizer(chars):
    """A Hugging Face fast tokenizer for our character vocabulary (WordLevel model, one token per character)."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

    tok = Tokenizer(models.WordLevel({c: i for i, c in enumerate(chars)}, unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tok.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(tokenizer_object=tok, eos_token=END, pad_token=END)


section("1. Convert our GPT to transformers' GPT2LMHeadModel")
ours, ckpt = GPT.load(ROOT / "runs" / "char_gpt.pt")
ours.eval()
vocab = CharVocab("".join(ckpt["chars"]))
hf = to_hf(ours, eos_id=vocab.stoi[END])
x = torch.tensor([vocab.encode("ROMEO:\nWhat light through yonder window")])
with torch.no_grad():
    diff = (hf(x).logits - ours(x, x)[0]).abs().max().item()
print(f"type: {type(hf).__name__} | max |logit difference| vs our model: {diff:.2e}")
print("-> same architecture, same weights: HF stores Linear weights transposed (Conv1D) — to_hf handles it.")

section("2. save_pretrained / from_pretrained: the format of every model on the Hub")
out_dir = ROOT / "runs" / "hf_char_gpt"
tok = char_tokenizer(ckpt["chars"])
hf.save_pretrained(out_dir)
tok.save_pretrained(out_dir)
print("files:", sorted(p.name for p in out_dir.iterdir()))
cfg = json.loads((out_dir / "config.json").read_text())
print("config.json (excerpt):", {k: cfg[k] for k in ["model_type", "n_layer", "n_head", "n_embd", "n_positions", "vocab_size"]})
model = AutoModelForCausalLM.from_pretrained(out_dir).eval()
tokenizer = AutoTokenizer.from_pretrained(out_dir)
print("AutoConfig says:", AutoConfig.from_pretrained(out_dir).model_type, "| AutoModelForCausalLM gave:",
      type(model).__name__)
print(f"parameters: {model.num_parameters():,} (HF counts the tied embedding once, like ours: "
      f"{ours.num_params(non_embedding=False):,})")

section("3. Tokenizer: text <-> ids, with the same ids as our CharVocab")
enc = tokenizer("To be, or not", return_tensors="pt")
print("input_ids:", enc.input_ids[0].tolist(), "| same as ours:", enc.input_ids[0].tolist() == vocab.encode("To be, or not"))
print("attention_mask:", enc.attention_mask[0].tolist(), "| decode:", repr(tokenizer.decode(enc.input_ids[0])))

section("4. generate(): greedy matches our implementation; sampling knobs have the same names")
prompt = tokenizer("ROMEO:\n", return_tensors="pt")
with torch.no_grad():
    g_hf = model.generate(**prompt, max_new_tokens=60, do_sample=False, pad_token_id=tokenizer.pad_token_id)
g_ours = ours.generate(prompt.input_ids, 60, temperature=0)
print("greedy, 60 tokens — identical to ours:", torch.equal(g_hf, g_ours))
print(repr(tokenizer.decode(g_hf[0])))
torch.manual_seed(0)
with torch.no_grad():
    s = model.generate(**prompt, max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.9, top_k=0,
                       pad_token_id=tokenizer.pad_token_id)
print("sampled (temperature=0.8, top_p=0.9):", repr(tokenizer.decode(s[0])))

section("5. Batching prompts of different lengths: padding side and attention_mask")
prompts = ["ROMEO:\n", "First Citizen:\nWe are"]
singles = []
with torch.no_grad():
    for p in prompts:
        e = tokenizer(p, return_tensors="pt")
        singles.append(tokenizer.decode(model.generate(**e, max_new_tokens=20, do_sample=False,
                                                       pad_token_id=tokenizer.pad_token_id)[0, e.input_ids.shape[1]:]))
    for side in ["right", "left"]:
        tokenizer.padding_side = side
        e = tokenizer(prompts, return_tensors="pt", padding=True)
        outs = model.generate(**e, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        batched = [tokenizer.decode(o[e.input_ids.shape[1]:]) for o in outs]
        print(f"padding_side={side:5s}: batched == one-at-a-time? {[b == s for b, s in zip(batched, singles)]}")
print("-> decoder-only models must be LEFT-padded for generation: new tokens are appended on the right, so")
print("   right padding puts pad tokens between the prompt and the continuation.")

section("6. Chat templates: our SFT model from 15.2, with a Jinja template stored in the tokenizer")
sft, _ = GPT.load(ROOT / "runs" / "sft_gpt.pt")
chat_model = to_hf(sft.eval(), eos_id=vocab.stoi[END])
chat_tok = char_tokenizer(ckpt["chars"])
chat_tok.chat_template = ("{% for m in messages %}{% if m['role'] == 'user' %}User: {{ m['content'] }}\n"
                          "{% else %}Bot: {{ m['content'] }}$\n{% endif %}{% endfor %}"
                          "{% if add_generation_prompt %}Bot: {% endif %}")
for msg in ["reverse tempest", "upper crown", "spell night"]:
    ids = chat_tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True)
    with torch.no_grad():
        out = chat_model.generate(**ids, max_new_tokens=20, do_sample=False, eos_token_id=chat_tok.eos_token_id,
                                  pad_token_id=chat_tok.pad_token_id)
    print(f"{chat_tok.decode(ids['input_ids'][0])!r:34s} -> {chat_tok.decode(out[0, ids['input_ids'].shape[1]:], skip_special_tokens=True)!r}")
print("-> apply_chat_template is how you talk to any instruct model: the template (in tokenizer_config.json)")
print("   must match the one used in SFT, or quality drops sharply.")

section("7. Looking inside: hidden states, attention, and the 'logit lens'")
text = "ROMEO:\nWhat light through yonder window br"
e = tokenizer(text, return_tensors="pt")
eager = AutoModelForCausalLM.from_pretrained(out_dir, attn_implementation="eager").eval()   # fused SDPA returns no weights
with torch.no_grad():
    o = eager(**e, output_hidden_states=True, output_attentions=True)
print("last hidden state already has ln_f applied:",
      torch.allclose(eager.lm_head(o.hidden_states[-1]), o.logits, atol=1e-5))
print(f"hidden_states: {len(o.hidden_states)} tensors of {tuple(o.hidden_states[0].shape)} (embeddings + one per block)")
print(f"attentions: {len(o.attentions)} tensors of {tuple(o.attentions[0].shape)} (batch, heads, query, key)")
print("logit lens — apply the final norm + output layer to each layer's residual stream; top guesses for the next char:")
with torch.no_grad():
    for i, h in enumerate(o.hidden_states):
        h = h if i == len(o.hidden_states) - 1 else model.transformer.ln_f(h)   # the last entry already has ln_f applied
        p = torch.softmax(model.lm_head(h[0, -1]), -1)
        top = p.topk(3)
        print(f"  after {'embeddings' if i == 0 else f'block {i}':10s}: " +
              "  ".join(f"{tokenizer.decode([int(j)])!r} {q:.2f}" for q, j in zip(top.values, top.indices)))
att = o.attentions[-1][0, :, -1]                                                # last block, last query position
print("last block, last position: each head's most-attended earlier character:",
      " ".join(repr(text[int(a.argmax())]) for a in att))

section("8. pipeline(): the one-liner")
gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(gen("JULIET:\n", max_new_tokens=40, do_sample=False)[0]["generated_text"])
