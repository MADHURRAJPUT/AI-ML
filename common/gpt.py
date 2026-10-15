"""A GPT in one file (Phase 3). Built in week 14, reused in weeks 15-16.

nanoGPT-style decoder-only transformer with GPT-2's architecture and parameter NAMES
(transformer.wte, transformer.h.0.attn.c_attn, ...), so weights convert to and from
Hugging Face's GPT2LMHeadModel (week 16). Adds an optional KV cache for fast generation.
"""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024          # maximum context length (learned positions, as in GPT-2)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.n_embd, self.dropout = cfg.n_head, cfg.n_embd, cfg.dropout
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, past_kv=None):
        """x: (B, T, C). past_kv: optional (k, v), each (B, H, T_past, d), from earlier calls.
        Returns (y, (k, v)) where k, v include the past — the updated cache."""
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        T_total = k.shape[2]
        drop = self.dropout if self.training else 0.0
        if T_total == T:                       # no cache: ordinary causal attention
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=True)
        elif T == 1:                           # one new token: it may attend to everything cached
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
        else:                                  # several new tokens after a cache: offset causal mask
            mask = torch.ones(T, T_total, dtype=torch.bool, device=x.device).tril(T_total - T)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y)), (k, v)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, past_kv=None):
        a, present = self.attn(self.ln_1(x), past_kv)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, present


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.config = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd, bias=cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight            # weight tying (GPT-2)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():                        # GPT-2: scale residual projections
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        return n - self.transformer.wpe.weight.numel() if non_embedding else n

    def forward(self, idx, targets=None, past_kv=None, use_cache=False):
        """idx: (B, T) token ids. Returns (logits, loss) or, with use_cache, (logits, loss, present_kv).
        past_kv: list with one (k, v) per layer from a previous call."""
        B, T = idx.shape
        past_len = 0 if past_kv is None else past_kv[0][0].shape[2]
        assert past_len + T <= self.config.block_size, f"sequence {past_len + T} > block_size {self.config.block_size}"
        pos = torch.arange(past_len, past_len + T, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        presents = []
        for i, block in enumerate(self.transformer.h):
            x, present = block(x, None if past_kv is None else past_kv[i])
            presents.append(present)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        else:
            logits, loss = self.lm_head(x[:, [-1], :]), None   # inference: only the last position is needed
        return (logits, loss, presents) if use_cache else (logits, loss)

    def configure_optimizers(self, weight_decay=0.1, lr=3e-4, betas=(0.9, 0.95)):
        """AdamW; decay only 2-D weights (matmuls, embeddings), never biases or LayerNorm gains."""
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [{"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
                  {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, use_cache=True,
                 generator=None, eos_id=None):
        """Autoregressive sampling. temperature=0 means greedy. With use_cache, past keys/values are
        reused, so each step processes ONE new token instead of the whole sequence."""
        self.eval()
        bs = self.config.block_size
        cache_ok = use_cache and idx.shape[1] + max_new_tokens <= bs
        past = None
        for _ in range(max_new_tokens):
            if cache_ok:
                inp = idx if past is None else idx[:, -1:]
                logits, _, past = self(inp, past_kv=past, use_cache=True)
            else:
                logits, _ = self(idx[:, -bs:])                       # crop to the context window
            next_id = sample_next(logits[:, -1, :], temperature, top_k, top_p, generator)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
        return idx

    def save(self, path, **extra):
        torch.save({"config": asdict(self.config), "model": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(GPTConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["model"])
        return model, ckpt


def sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    """logits: (B, V) -> (B, 1) sampled ids. temperature 0 = argmax."""
    if temperature == 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None:                                          # nucleus: smallest set with mass >= top_p
        sorted_logits, order = torch.sort(logits, descending=True)
        cum = F.softmax(sorted_logits, -1).cumsum(-1)
        remove = cum - F.softmax(sorted_logits, -1) >= top_p     # keep tokens until the mass BEFORE them reaches p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)
    return torch.multinomial(F.softmax(logits, -1), 1, generator=generator)


# ------------------------------------------------------------------ Hugging Face interop (week 16) ----
_TRANSPOSED = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


def to_hf(model, eos_id=None):
    """Convert our GPT into transformers' GPT2LMHeadModel (HF stores these matrices as Conv1D: transposed).
    eos_id: id of the end-of-text token in OUR vocabulary (GPT-2's 50256 by default for its vocab)."""
    from transformers import GPT2Config, GPT2LMHeadModel

    c = model.config
    if eos_id is None and c.vocab_size == 50257:
        eos_id = 50256
    hf = GPT2LMHeadModel(GPT2Config(vocab_size=c.vocab_size, n_positions=c.block_size, n_embd=c.n_embd,
                                    n_layer=c.n_layer, n_head=c.n_head, activation_function="gelu_new",
                                    resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
                                    bos_token_id=eos_id, eos_token_id=eos_id))
    sd, hf_sd = model.state_dict(), hf.state_dict()
    with torch.no_grad():
        for k, v in sd.items():
            if k not in hf_sd:
                continue
            hf_sd[k].copy_(v.t() if k.endswith(_TRANSPOSED) else v)
    return hf.eval()


def from_hf(hf_model):
    """Load a GPT2LMHeadModel (e.g. GPT2LMHeadModel.from_pretrained('gpt2')) into our GPT class."""
    hc = hf_model.config
    model = GPT(GPTConfig(vocab_size=hc.vocab_size, block_size=hc.n_positions, n_layer=hc.n_layer,
                          n_head=hc.n_head, n_embd=hc.n_embd))
    sd, hf_sd = model.state_dict(), hf_model.state_dict()
    with torch.no_grad():
        for k in sd:
            if k == "lm_head.weight":
                sd[k].copy_(hf_sd["transformer.wte.weight"]); continue
            sd[k].copy_(hf_sd[k].t() if k.endswith(_TRANSPOSED) else hf_sd[k])
    return model
