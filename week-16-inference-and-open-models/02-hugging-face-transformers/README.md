# 16.2 Hugging Face `transformers`: loading, inspecting and sampling open models

Almost every open-weight model (LLaMA, Qwen, Mistral, Gemma, Phi, DeepSeek)
ships in the same format and loads with the same five calls. This lesson
teaches that workflow on **our own models converted to HF format**, so it runs
offline and you can check every number against code you wrote. Then
`real_models.py` repeats it on real models from the Hub.

```bash
python hf_tour.py       # ~15 s; needs runs/char_gpt.pt (14.2) and runs/sft_gpt.pt (15.2)
python real_models.py   # on YOUR machine: downloads gpt2 + Qwen2.5-0.5B-Instruct (not run in the course build: no Hub access)
```

## The workflow

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
cfg   = AutoConfig.from_pretrained(name)                         # architecture, from config.json
tok   = AutoTokenizer.from_pretrained(name)                      # tokenizer.json + tokenizer_config.json (incl. chat template)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)   # weights from *.safetensors
ids   = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
out   = model.generate(**ids, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)
```

## What hf_tour.py shows

| # | step | result |
|---|---|---|
| 1 | `to_hf(our_gpt)` → `GPT2LMHeadModel` | max logit difference 1.6e-5: **the same model** (HF stores Linear weights transposed, as `Conv1D`) |
| 2 | `save_pretrained` → `from_pretrained` | `config.json`, `generation_config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`: exactly what you download from the Hub |
| 3 | a fast tokenizer for our character vocabulary | same ids as `CharVocab`; returns an `attention_mask` |
| 4 | `generate(do_sample=False)` | **identical** to our `GPT.generate(temperature=0)` for 60 tokens; `temperature`, `top_k`, `top_p` mean exactly what they meant in 14.4 |
| 5 | batched generation with padding | right padding: `[False, True]` (the padded prompt went wrong); **left padding: `[True, True]`** |
| 6 | `apply_chat_template` with our SFT model | `'User: reverse tempest\nBot: '` → `'tsepmet'` (and `upper crown → 'CROWNW'`: it's the same imperfect model from 15.2) |
| 7 | `output_hidden_states`, `output_attentions`, logit lens | see below |
| 8 | `pipeline("text-generation")` | the one-liner |

**Two gotchas you will hit with real models:**
* **Pad on the left for generation.** A decoder appends new tokens on the right. With right padding, pad tokens sit between the prompt and the continuation, and the shorter prompt's output was wrong.
* **The chat template must match the one used in training.** `apply_chat_template` reads it from `tokenizer_config.json`. Writing the prompt string by hand is the most common cause of a "the model is dumb" bug.

## Looking inside a model

Pass `output_hidden_states=True` to get the residual stream after every block.
Passing `output_attentions=True` needs `attn_implementation="eager"`, because
the fused SDPA kernel never materialises the attention weights (13.1).

**The logit lens** (nostalgebraist, 2020): apply the final norm and output
layer to each *intermediate* residual stream, and read off what the model
"would predict" at that depth. For `"...yonder window br"`:

| after | top guesses |
|---|---|
| embeddings | `'r'` 1.00 |
| block 1 | `'r'` 0.65, `'d'` 0.10, `'e'` 0.10 |
| block 2 | `'r'` 0.30, `'e'` 0.29, `'i'` 0.14 |
| block 3 | `'e'` 0.29, `'i'` 0.26, `'o'` 0.26 |
| block 4 (output) | `'o'` 0.35, `'e'` 0.33, `'i'` 0.19 |

Straight after the embeddings, the model "predicts" the **current**
character, `'r'`. With tied weights, the embedding of `r` scores highest
against itself. Block by block, the residual stream turns into a prediction of
the **next** character (`'o'`, `'e'`, `'i'`: "bro…", "bre…", "bri…"). You can
watch the computation happen across depth. Doing this on a real model
(`real_models.py`, section 4) is a first step into interpretability.

## real_models.py (on your machine)

1. **GPT-2 into our class:** `from_hf(GPT2LMHeadModel.from_pretrained("gpt2"))`. The logits should match HF's, and the parameter count should match week 13.4's prediction of 124 439 808.
2. **Read a modern architecture** from `config.json`: layers, hidden size, `num_key_value_heads` (GQA), `rope_theta`, vocabulary size. From these, compute the KV cache per token yourself.
3. **Chat** through `apply_chat_template`. Print the templated string to see the special tokens the model was trained on.
4. **Logit lens** on a 0.5 B model.

## Exercises

1. In `hf_tour.py`, sample 5 continuations with `num_return_sequences=5, do_sample=True`. Then try `repetition_penalty=1.3`. What changes?
2. Using `output_attentions`, find a head in our char GPT that mostly attends to the previous character. (Average `att[h, t, t-1]` over a long text.)
3. On your machine: load `Qwen/Qwen2.5-0.5B-Instruct` and an 8 B model in 4-bit (`BitsAndBytesConfig(load_in_4bit=True)` on a CUDA GPU, or a GGUF in llama.cpp/Ollama). Measure tokens per second and compare with 16.1's arithmetic.

<details><summary>Hints</summary>

2. In small char models one head is often a strong "previous token" head (> 0.5 average weight on t−1). Look across layers 0–1 first.
3. On a laptop CPU, 0.5 B in bf16 runs at roughly tens of tokens/s. Memory bandwidth divided by model size gives you the order of magnitude, just as in 16.1.

</details>
