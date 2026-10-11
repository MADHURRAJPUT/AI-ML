# 8.3 `nn.Module` — how models own their parameters

`nn.Module` is a container that does bookkeeping: it registers parameters,
sub-modules and buffers when you assign them as attributes, so that
`model.parameters()`, `model.to(device)`, `model.state_dict()` and
`model.train()/eval()` reach everything automatically.

## Anatomy

```python
class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, p_drop=0.1):
        super().__init__()                       # must come first
        self.fc1 = nn.Linear(d_in, d_hidden)     # sub-module -> its params are registered
        self.drop = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):                        # define forward only; autograd does backward
        return self.fc2(self.drop(F.relu(self.fc1(x))))

model = MLP(784, 256, 10)
logits = model(x)                                # call the module, not .forward() (hooks run)
```

What gets registered:

| Thing | How | In `parameters()`? | In `state_dict()`? | Moves with `.to()`? |
|---|---|---|---|---|
| `nn.Parameter` attribute | `self.w = nn.Parameter(t)` | ✅ | ✅ | ✅ |
| sub-module attribute | `self.fc = nn.Linear(...)` | ✅ (its params) | ✅ | ✅ |
| **buffer** | `self.register_buffer("mask", t)` | ❌ | ✅ (unless `persistent=False`) | ✅ |
| plain tensor attribute | `self.t = t` | ❌ | ❌ | ❌ ← common bug |
| list of modules | `nn.ModuleList([...])` / `nn.ModuleDict` | ✅ | ✅ | ✅ |
| Python list of modules | `[nn.Linear(...)]` | ❌ ← common bug | ❌ | ❌ |

Buffers are state that isn't trained: BatchNorm's running mean/variance, a
causal attention mask, positional-encoding tables.

## Building blocks

- `nn.Sequential(...)` — layers applied in order; fine for simple stacks.
- `nn.ModuleList` — when `forward` needs a loop or skip connections (transformer blocks).
- `nn.Linear`, `nn.Embedding` (a lookup table — Phase 3), `nn.Conv2d` (week 10), `nn.LayerNorm`, `nn.Dropout`, activations…
- `torch.nn.functional` (`F`) — stateless versions (`F.relu`, `F.cross_entropy`). Rule of thumb: things with parameters or train/eval behaviour are modules; pure functions can be `F.*`.

## `train()` vs `eval()`

`model.train()` / `model.eval()` set `self.training` recursively. Only some
layers care: **Dropout** (on in train, identity in eval) and **BatchNorm**
(batch statistics in train, running averages in eval). Forgetting `eval()` makes
predictions random (dropout) or batch-dependent (BN).

## `state_dict` — the canonical way to save models

```python
torch.save(model.state_dict(), "model.pt")     # an ordered dict: name -> tensor
model = MLP(784, 256, 10)                      # rebuild the architecture in code
model.load_state_dict(torch.load("model.pt"))  # strict=True checks every key matches
```

Save the `state_dict`, not the pickled model object — pickles break when code
moves. Hugging Face `safetensors` files are state_dicts in a safe, memory-mappable
format; LoRA adapters (Phase 5) are a *small* state_dict loaded on top of a big one.

## Initialisation

Each built-in layer has a default init (`nn.Linear` uses Kaiming-uniform with
a=√5). Override with `nn.init.*` inside `__init__` or via `model.apply(fn)`.
Week 9.1 explains which init to choose and why.

## Run the demo

```bash
python modules.py
```

Builds `Linear` from scratch with `nn.Parameter` (and checks it equals
`nn.Linear`), shows registration rules and the plain-list bug, buffers, counting
and grouping parameters, `train()`/`eval()` with dropout, `state_dict`
save/load round-trip, custom init via `apply`, and a forward hook capturing
activations.

## Exercises

1. Write `class Residual(nn.Module)` that wraps any module `f` and returns `x + f(x)`.
2. Count parameters of `nn.Embedding(50257, 768)`. Why is it the biggest single tensor in GPT-2 small?
3. Create two parameter groups: weights get `weight_decay=0.1`, biases and norm parameters get 0. (This is the standard transformer AdamW setup.)

<details><summary>Answers</summary>

1. `def __init__(self, f): super().__init__(); self.f = f` / `def forward(self, x): return x + self.f(x)`.
2. 50257 × 768 ≈ 38.6 M — one vector per vocabulary token; GPT-2 small has 124 M total.
3. `decay = [p for n, p in model.named_parameters() if p.ndim >= 2]; no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]; AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}])`.

</details>
