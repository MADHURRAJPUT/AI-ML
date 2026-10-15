"""Week 12.4 — where tokenisation bites: numbers, whitespace, case, spelling, languages, cost."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.bpe import shakespeare_tokenizer  # noqa: E402

tok = shakespeare_tokenizer(1024)


def section(title):
    print(f"\n=== {title} ===")


def show(s):
    ids = tok.encode(s)
    return f"{s!r:28s} -> {len(ids):2d} tokens {tok.token_strings(ids)}"


section("Numbers are chunked arbitrarily -> arithmetic is hard")
for s in ["7", "42", "127", "1234", "12345", "3.14159", "2026-10-15"]:
    print(show(s))
print("-> here every digit is its own token: Shakespeare has almost no numbers to learn merges from.")
print("   GPT-4 splits '1234' into '123' + '4', GPT-2 learned arbitrary multi-digit tokens. Either way the")
print("   model sees pieces, not quantities — carrying across chunk boundaries must be learned the hard way.")

section("Whitespace and case create different tokens for 'the same' word")
for s in ["hello", " hello", "Hello", " Hello", "HELLO", "  hello", "hello "]:
    print(show(s))
print("-> a prompt ending in a space ('The answer is ') leaves the model a token boundary it rarely saw in")
print("   training (the space normally belongs to the NEXT word). Trailing whitespace degrades outputs.")

section("Spelling: the model sees chunks, not letters")
for w in [" strawberry", " Mississippi", " unbelievable"]:
    ids = tok.encode(w)
    print(f"{w!r:16s} -> {tok.token_strings(ids)}   letters r: {w.count('r')}, s: {w.count('s')}")
print("-> 'how many r's in strawberry?' asks about characters the model never directly observes.")

section("Languages: the same sentence costs very different numbers of tokens")
sentences = {
    "English": "The weather is beautiful today and I want to go for a walk in the park.",
    "French": "Le temps est magnifique aujourd'hui et je veux aller me promener dans le parc.",
    "German": "Das Wetter ist heute wunderschön und ich möchte im Park spazieren gehen.",
    "Spanish": "Hoy hace un tiempo precioso y quiero ir a dar un paseo por el parque.",
    "Hindi": "आज मौसम बहुत सुंदर है और मैं पार्क में टहलने जाना चाहता हूँ।",
    "Japanese": "今日は天気がとても良いので、公園を散歩したいです。",
    "Chinese": "今天天气很好，我想去公园散步。",
}
base = len(tok.encode(sentences["English"]))
print(f"{'language':9s} {'chars':>5s} {'bytes':>6s} {'tokens':>7s} {'vs English':>11s}")
for lang, s in sentences.items():
    n = len(tok.encode(s))
    print(f"{lang:9s} {len(s):5d} {len(s.encode()):6d} {n:7d} {n / base:10.1f}x")
print("-> an English-trained tokenizer falls back to raw bytes for other scripts: Hindi costs several")
print("   times more tokens for the same meaning — more money, more latency, less fits in the context.")
print("   Big multilingual vocabularies (GPT-4o, LLaMA-3, Gemma) shrink, but rarely remove, this gap.")

section("Code: indentation and symbols")
code = "def add(a, b):\n    return a + b\n"
print(show(code))

section("Estimating cost before you build")


def estimate(chars_in, chars_out, chars_per_token, price_in_per_m, price_out_per_m, requests):
    t_in, t_out = chars_in / chars_per_token, chars_out / chars_per_token
    per_req = t_in * price_in_per_m / 1e6 + t_out * price_out_per_m / 1e6
    return t_in, t_out, per_req, per_req * requests


# ILLUSTRATIVE prices only (USD per 1M tokens) — look up your provider's current price list.
t_in, t_out, per_req, monthly = estimate(chars_in=12_000, chars_out=1_500, chars_per_token=4.0,
                                         price_in_per_m=3.0, price_out_per_m=15.0, requests=200_000)
print(f"RAG answer: ~{t_in:,.0f} input tokens (question + retrieved context), ~{t_out:,.0f} output tokens")
print(f"at $3 / $15 per 1M in/out tokens (illustrative): ${per_req:.4f} per request, ${monthly:,.0f} for 200k requests")
print("-> input tokens usually dominate count; output tokens are pricier per token and SLOWER (each one is a")
print("   sequential decode step — week 16). Prompt caching and shorter contexts cut input cost (Phase 4).")
print("   Rule of thumb: ~4 characters ≈ 1 token for English with a ~100k vocabulary; measure your own data.")
