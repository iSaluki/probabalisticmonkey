"""Probabilistic Monkey model definition (PyTorch, used for training).

A decoder-only transformer with:
  * RMSNorm pre-normalisation
  * rotary position embeddings (RoPE)
  * multi-head causal self-attention
  * a sparse Mixture-of-Experts SwiGLU feed-forward block (top-1 routing,
    Switch-Transformer style) plus one always-on shared expert

The MoE layout gives the model a large total parameter count (>100M) while
keeping the compute per token small, which is what makes it fast on a CPU.
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 16384
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    n_experts: int = 8          # routed experts per layer
    d_expert: int = 1024        # hidden size of each routed expert
    d_shared: int = 1024        # hidden size of the always-on shared expert
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    aux_loss_coef: float = 0.01

    def to_dict(self):
        return asdict(self)


def rope_tables(cfg, device=None):
    hd = cfg.d_model // cfg.n_heads
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, hd, 2, device=device).float() / hd))
    t = torch.arange(cfg.max_seq_len, device=device).float()
    freqs = torch.outer(t, inv)  # T, hd/2
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: B, H, T, hd   (rotate-half convention on interleaved halves)
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None]
    sin = sin[None, None]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MoE(nn.Module):
    """Top-1 routed experts + a shared expert, SwiGLU activations.

    Training uses fixed-capacity dispatch (Switch Transformer): every expert
    processes exactly `capacity` token slots so the expert maths is one big
    batched matmul. Tokens that overflow an expert's capacity skip the routed
    expert (they still get the shared expert and the residual path).
    """

    def __init__(self, cfg):
        super().__init__()
        d, E, h = cfg.d_model, cfg.n_experts, cfg.d_expert
        self.E = E
        self.capacity_factor = 1.25
        self.router = nn.Linear(d, E, bias=False)
        self.w_in = nn.Parameter(torch.empty(E, d, 2 * h))   # fused gate+up
        self.w_out = nn.Parameter(torch.empty(E, h, d))
        self.shared_in = nn.Linear(d, 2 * cfg.d_shared, bias=False)
        self.shared_out = nn.Linear(cfg.d_shared, d, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        E = self.E
        xf = x.reshape(-1, C)
        N = xf.shape[0]
        g, u = self.shared_in(xf).chunk(2, dim=-1)
        out = self.shared_out(F.silu(g) * u)

        probs = self.router(xf).float().softmax(-1)
        top_p, top_i = probs.max(-1)
        onehot = F.one_hot(top_i, E)
        aux = E * (onehot.float().mean(0) * probs.mean(0)).sum()   # load balancing

        # no token dropping at evaluation time
        cap = int(math.ceil(N / E * self.capacity_factor)) if self.training else N
        pos = (onehot.cumsum(0) - 1).gather(1, top_i[:, None]).squeeze(1)
        keep = pos < cap
        slot = torch.where(keep, top_i * cap + pos, torch.full_like(pos, E * cap))
        buf = xf.new_zeros(E * cap + 1, C).index_put((slot,), xf)
        h = buf[:-1].view(E, cap, C)
        ge, ue = torch.bmm(h, self.w_in.to(h.dtype)).chunk(2, dim=-1)
        y = torch.bmm(F.silu(ge) * ue, self.w_out.to(h.dtype)).reshape(E * cap, C)
        y = torch.cat([y, y.new_zeros(1, C)], 0)
        gate = (top_p * keep).to(y.dtype).unsqueeze(-1)
        out = out + y[slot] * gate
        return out.view(B, T, C), aux


# Precision of the output-layer matmul; train.py switches it to float32 on CPUs
# without fast bfloat16 support.
COMPUTE_DTYPE = torch.bfloat16


def _lm_loss(x, w, targets, mask):
    logits = (x.to(COMPUTE_DTYPE) @ w.to(COMPUTE_DTYPE).T).float()
    per = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
    m = mask.reshape(-1).float()
    return (per * m).sum() / m.sum().clamp(min=1)


lm_loss = _lm_loss


def enable_compile():
    """Compile the dense hot spots (loss head + attention) for faster CPU training."""
    global lm_loss
    lm_loss = torch.compile(_lm_loss)
    Block.forward = torch.compile(Block.forward)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = nn.RMSNorm(cfg.d_model, eps=1e-5)
        self.attn = Attention(cfg)
        self.n2 = nn.RMSNorm(cfg.d_model, eps=1e-5)
        self.moe = MoE(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        m, aux = self.moe(self.n2(x))
        return x + m, aux


class Monkey(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.RMSNorm(cfg.d_model, eps=1e-5)
        cos, sin = rope_tables(cfg)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self._init()

    def _init(self):
        std = 0.02
        for name, p in self.named_parameters():
            if p.dim() < 2:
                continue
            nn.init.normal_(p, 0.0, std)
            # scale residual output projections down with depth (GPT-2 style)
            if name.endswith(("proj.weight", "shared_out.weight", "w_out")):
                with torch.no_grad():
                    p.mul_(1 / math.sqrt(2 * self.cfg.n_layers))

    def forward(self, idx, targets=None, loss_mask=None):
        x = self.embed(idx)
        aux_total = 0.0
        for b in self.blocks:
            x, aux = b(x, self.cos, self.sin)
            aux_total = aux_total + aux
        x = self.norm(x)
        if targets is None:
            return x @ self.embed.weight.T.to(x.dtype)  # tied embeddings
        if loss_mask is None:
            loss_mask = torch.ones_like(targets)
        loss = lm_loss(x, self.embed.weight, targets, loss_mask)
        aux_total = aux_total / len(self.blocks)
        return None, loss, aux_total

    def param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        c = self.cfg
        per_expert = 3 * c.d_model * c.d_expert
        inactive = c.n_layers * (c.n_experts - 1) * per_expert
        return total, total - inactive
