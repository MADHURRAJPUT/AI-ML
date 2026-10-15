"""Week 16.2 — the same workflow on real open-weight models from the Hugging Face Hub. RUN THIS ON YOUR MACHINE.

    pip install -U transformers accelerate
    python real_models.py                     # gpt2 (500 MB) + Qwen2.5-0.5B-Instruct (1 GB); CPU is fine

Note: the course's build environment has no access to huggingface.co, so this file was NOT executed there
(hf_tour.py runs the same calls on our own converted models). The config/model calls were checked against a
randomly initialised Qwen2 with transformers 5.19. If an API has moved in your version, the error usually names
the replacement.
"""
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.gpt import from_hf  # noqa: E402


def section(title):
    print(f"\n=== {title} ===")


section("1. GPT-2 (124 M): load it into OUR GPT class and check the logits match")
tok = AutoTokenizer.from_pretrained("gpt2")
hf = AutoModelForCausalLM.from_pretrained("gpt2").eval()
ours = from_hf(hf).eval()
ids = tok("The meaning of life is", return_tensors="pt").input_ids
with torch.no_grad():
    print("max |logit diff|:", (hf(ids).logits - ours(ids, ids)[0]).abs().max().item())
print("parameters:", f"{sum(p.numel() for p in ours.parameters()):,}", "(week 13.4 predicted 124,439,808)")
out = ours.generate(ids, 30, temperature=0.8, top_k=50)
print(tok.decode(out[0]))

section("2. A modern instruct model: read its architecture from config.json")
name = "Qwen/Qwen2.5-0.5B-Instruct"
cfg = AutoConfig.from_pretrained(name)
for key in ["model_type", "num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads",
            "num_key_value_heads", "vocab_size", "max_position_embeddings", "tie_word_embeddings"]:
    print(f"  {key:24s} {getattr(cfg, key, None)}")
rope = getattr(cfg, "rope_theta", None) or (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta")
print(f"  {'rope_theta':24s} {rope}   (transformers 5 keeps it in cfg.rope_parameters)")
head_dim = cfg.hidden_size // cfg.num_attention_heads
kv_bytes = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * 2
print(f"-> GQA with {cfg.num_attention_heads} query / {cfg.num_key_value_heads} KV heads; KV cache "
      f"{kv_bytes / 1024:.1f} KB per token (bf16), {kv_bytes * 32768 / 2**30:.2f} GiB for 32 k tokens")

section("3. Chat with it: apply_chat_template + generate")
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).eval()
print(f"parameters: {model.num_parameters():,}")
messages = [{"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Explain the KV cache in two sentences."}]
prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("the template turns messages into this exact string:\n" + prompt_text)
inputs = tok(prompt_text, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=120, do_sample=True, temperature=0.7, top_p=0.9)
print(tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))

section("4. Next-token distribution and the logit lens on a real model")
inputs = tok("The capital of France is", return_tensors="pt")
with torch.no_grad():
    o = model(**inputs, output_hidden_states=True)
p = torch.softmax(o.logits[0, -1].float(), -1)
print("top next tokens:", [(tok.decode([int(i)]), round(float(q), 3)) for q, i in zip(*p.topk(5))])
norm = model.model.norm                                   # LLaMA/Qwen naming: model.model.norm, model.lm_head
with torch.no_grad():
    for i in range(0, len(o.hidden_states), 4):
        h = o.hidden_states[i][0, -1]
        h = h if i == len(o.hidden_states) - 1 else norm(h)
        top = int(model.lm_head(h).argmax())
        print(f"  layer {i:2d}: {tok.decode([top])!r}")
