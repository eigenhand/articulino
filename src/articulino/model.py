"""The acoustic model: text encoder, voice encoder, duration predictor and the 4-step DiT (student).

Shapes use B = batch, N = characters (incl. the prompt placeholder at index 0), R = reference frames,
L = latent frames (25 Hz), P = prompt frames, D = model width.

Everything here is plain PyTorch in float32 and follows the trained network exactly:

* LayerNorm eps 1e-5, exact (erf) GELU, SiLU in the time and voice-vector MLPs.
* Attention: one ``kv_proj`` matrix (keys first, then values), heads split after the projection, softmax(q·kᵀ/√d)·v.
* Rotary position embedding on interleaved pairs (even/odd feature indices), self-attention only.
* The DiT continues from a clean voice prompt. The prompt frames attend only to themselves, so their keys and values
  are computed once per voice (:meth:`DiT.prompt_cache`) and every sentence only computes its new frames
  (:meth:`DiT.velocity`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ArticulinoConfig, DiTConfig, DurationConfig, TextEncoderConfig, VoiceEncoderConfig

KV = tuple[torch.Tensor, torch.Tensor]
_TIME_BASE = 10000.0


# --------------------------------------------------------------------------------------------- building blocks
def rope_table(length: int, head_dim: int, base: float = 10000.0, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin tables (length, head_dim/2) for positions 0..length-1."""
    freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angle = torch.outer(torch.arange(length, device=device).float(), freq)
    return angle.cos(), angle.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate interleaved pairs (x[..., 0::2], x[..., 1::2]) of x (B, H, L, head_dim)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """softmax(q·kᵀ/√d)·v over heads; q (B, H, L, d), k/v (B, H, S, d) -> (B, L, H·d).

    ``mask`` (B, S) bool marks valid keys; invalid keys get -1e9 before the softmax."""
    b, h, l, d = q.shape
    att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
    if mask is not None:
        att = att.masked_fill(~mask[:, None, None, :], -1e9)
    return torch.matmul(att.softmax(-1), v).transpose(1, 2).reshape(b, l, h * d)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, context_dim: int | None = None):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(context_dim or dim, 2 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def keys_values(self, context: torch.Tensor) -> KV:
        """Keys and values (B, H, S, head_dim) of ``context`` (B, S, context_dim)."""
        b, s, _ = context.shape
        k, v = self.kv_proj(context).view(b, s, 2, self.heads, -1).permute(2, 0, 3, 1, 4)
        return k, v

    def queries(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        return self.q_proj(x).view(b, l, self.heads, -1).transpose(1, 2)

    def forward(self, x, context=None, mask=None, rope=None, kv: KV | None = None):
        q = self.queries(x)
        k, v = kv if kv is not None else self.keys_values(x if context is None else context)
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        return self.out_proj(attend(q, k, v, mask))


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.up = nn.Linear(dim, dim * mult)
        self.down = nn.Linear(dim * mult, dim)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class MLP(nn.Module):
    """Linear -> SiLU -> Linear."""

    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))


class EncoderLayer(nn.Module):
    """Pre-norm transformer layer with rotary self-attention (text and voice encoders)."""

    def __init__(self, dim: int, heads: int, ff_mult: int = 4, eps: float = 1e-5):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, heads)
        self.ff_norm = nn.LayerNorm(dim, eps=eps)
        self.ff = FeedForward(dim, ff_mult)

    def forward(self, x, mask, rope):
        x = x + self.attn(self.attn_norm(x), mask=mask, rope=rope)
        return x + self.ff(self.ff_norm(x))


# --------------------------------------------------------------------------------------------- text encoder
class TextEncoder(nn.Module):
    """Character ids + language ids -> features per character (B, N, 512)."""

    def __init__(self, vocab_size: int, cfg: TextEncoderConfig):
        super().__init__()
        self.heads = cfg.heads
        self.char_embedding = nn.Embedding(vocab_size, cfg.dim, padding_idx=0)
        self.language_embedding = nn.Embedding(2, cfg.dim)
        self.layers = nn.ModuleList(EncoderLayer(cfg.dim, cfg.heads, cfg.ff_mult, cfg.layer_norm_eps)
                                    for _ in range(cfg.layers))
        self.final_norm = nn.LayerNorm(cfg.dim, eps=cfg.layer_norm_eps)

    def forward(self, ids: torch.Tensor, language: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ids (B, N); language (B,) for the whole text or (B, N) per character; mask (B, N) bool, True = real
        character (the prompt placeholder is False)."""
        lang = self.language_embedding(language)
        x = self.char_embedding(ids) + (lang[:, None] if language.dim() == 1 else lang)
        rope = rope_table(x.shape[1], x.shape[2] // self.heads, device=x.device)
        for layer in self.layers:
            x = layer(x, mask, rope)
        return self.final_norm(x) * mask[..., None]


# --------------------------------------------------------------------------------------------- voice encoder
class PoolLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int = 4):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_mult)


class VoiceEncoder(nn.Module):
    """Normalized reference latents (B, R, 96) -> 32 voice tokens (B, 32, 768) and a voice vector (B, 768)."""

    def __init__(self, cfg: VoiceEncoderConfig):
        super().__init__()
        self.heads = cfg.heads
        self.input_proj = nn.Linear(cfg.input_dim, cfg.dim)
        self.layers = nn.ModuleList(EncoderLayer(cfg.dim, cfg.heads, cfg.ff_mult) for _ in range(cfg.layers))
        self.queries = nn.Parameter(torch.zeros(cfg.num_tokens, cfg.dim))
        self.pool = nn.ModuleList(PoolLayer(cfg.dim, cfg.heads, cfg.ff_mult) for _ in range(cfg.pool_layers))
        self.final_norm = nn.LayerNorm(cfg.dim)
        self.token_proj = nn.Linear(cfg.dim, cfg.out_dim)
        self.vector_mlp = MLP(cfg.dim, cfg.out_dim, cfg.out_dim)

    def forward(self, latents: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(latents)
        rope = rope_table(x.shape[1], x.shape[2] // self.heads, device=x.device)
        for layer in self.layers:
            x = layer(x, mask, rope)
        q = self.queries[None].expand(x.shape[0], -1, -1)
        for p in self.pool:
            q = q + p.attn(p.attn_norm(q), context=x, mask=mask)
            q = q + p.ff(p.ff_norm(q))
        q = self.final_norm(q)
        return self.token_proj(q), self.vector_mlp(q.mean(1))


# --------------------------------------------------------------------------------------------- durations
class DurationPredictor(nn.Module):
    """Character features + voice vector -> log duration per character, in 10 ms units."""

    def __init__(self, cfg: DurationConfig):
        super().__init__()
        c, k = cfg.channels, cfg.kernel_size
        self.voice_proj = nn.Linear(cfg.voice_dim, cfg.input_dim)
        self.convs = nn.ModuleList([nn.Conv1d(cfg.input_dim, c, k, padding=k // 2),
                                    nn.Conv1d(c, c, k, padding=k // 2), nn.Conv1d(c, c, k, padding=k // 2)])
        self.norms = nn.ModuleList(nn.LayerNorm(c) for _ in range(3))
        self.out_proj = nn.Linear(c, 1)

    def forward(self, h: torch.Tensor, voice_vector: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask[..., None]
        x = (h + self.voice_proj(voice_vector)[:, None]) * m
        for conv, norm in zip(self.convs, self.norms):
            x = norm(F.relu(conv(x.transpose(1, 2))).transpose(1, 2)) * m
        return self.out_proj(x).squeeze(-1)


def expand_to_frames(h: torch.Tensor, durations: torch.Tensor, frames: int, units_per_frame: int = 4) -> torch.Tensor:
    """Spread character features over latent frames.

    h (B, N, C), durations (B, N) in 10 ms units (may be fractional) -> (B, frames, C). The weight of character n
    on frame j is the overlap of [start_n, end_n) with [4j, 4j + 4), divided by 4."""
    end = durations.cumsum(1)
    start = end - durations
    lo = torch.arange(frames, device=h.device)[None, :, None] * units_per_frame
    w = (torch.minimum(end[:, None, :], lo + units_per_frame) - torch.maximum(start[:, None, :], lo)).clamp(min=0)
    return torch.bmm(w.to(h.dtype) / units_per_frame, h)


# --------------------------------------------------------------------------------------------- DiT
class AdaNorm(nn.Module):
    """LayerNorm without weights (eps 1e-5), then x·(1 + scale) + shift with [scale, shift] = Linear(c)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, 2 * dim)

    def modulation(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """c (B, D) -> scale, shift (B, 1, D): the same for every frame."""
        scale, shift = self.proj(c)[:, None].chunk(2, -1)
        return scale, shift

    def forward(self, x: torch.Tensor, mod: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        scale, shift = mod
        return F.layer_norm(x, (self.dim,), eps=1e-5) * (1 + scale) + shift


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int = 4):
        super().__init__()
        self.attn_norm = AdaNorm(dim)
        self.attn = Attention(dim, heads)
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attn = Attention(dim, heads, context_dim=dim)
        self.ff_norm = AdaNorm(dim)
        self.ff = FeedForward(dim, ff_mult)

    def forward(self, x, c, rope, cross_kv: KV, prompt_kv: KV | None = None, keep: list | None = None):
        """x (B, L, D); c (B, D) condition (time + voice); rope for the positions of x; cross_kv: keys/values of
        the voice tokens; prompt_kv: keys/values of the prompt frames in front of x (attended without mask);
        keep: list that receives this block's own (k, v) (to build the prompt cache)."""
        xn = self.attn_norm(x, self.attn_norm.modulation(c))
        q = apply_rope(self.attn.queries(xn), *rope)
        k, v = self.attn.keys_values(xn)
        k = apply_rope(k, *rope)
        if keep is not None:
            keep.append((k, v))
        if prompt_kv is not None:
            k, v = torch.cat([prompt_kv[0].to(k.dtype), k], 2), torch.cat([prompt_kv[1].to(v.dtype), v], 2)
        x = x + self.attn.out_proj(attend(q, k, v))
        x = x + self.cross_attn(self.cross_norm(x), kv=cross_kv)
        return x + self.ff(self.ff_norm(x, self.ff_norm.modulation(c)))


@dataclass
class VoiceState:
    """Everything that depends only on the voice, computed once by :meth:`AcousticModel.prepare_voice`."""

    latents: torch.Tensor           # (1, R, 96) normalized reference latents
    tokens: torch.Tensor            # (1, 32, 768) voice tokens
    vector: torch.Tensor            # (1, 768) voice vector
    prompt: torch.Tensor            # (1, P, 96) the first P reference frames, continued from
    dit_vector: torch.Tensor        # (1, 384) voice vector in DiT width
    cross_kv: list[KV]              # per DiT block: keys/values of the voice tokens
    prompt_kv: list[KV]             # per DiT block: keys/values of the prompt frames (the voice cache)

    @property
    def prompt_frames(self) -> int:
        return self.prompt.shape[1]


class DiT(nn.Module):
    """Flow-matching velocity v(x_t, t) on normalized 96-dim latents (student: 8 blocks, D = 384, 6 heads)."""

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        self.heads = cfg.heads
        d = cfg.dim
        self.text_proj = nn.Linear(cfg.text_dim, d)
        self.voice_token_proj = nn.Linear(cfg.voice_dim, d)
        self.voice_vector_proj = nn.Linear(cfg.voice_dim, d)
        self.input_proj = nn.Linear(2 * cfg.latent_dim + 1, d)
        self.time_mlp = MLP(cfg.time_embedding_dim, d, d)
        self.blocks = nn.ModuleList(DiTBlock(d, cfg.heads, cfg.ff_mult) for _ in range(cfg.layers))
        self.final_norm = AdaNorm(d)
        self.output_proj = nn.Linear(d, cfg.latent_dim)

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal features of t (B,) -> (B, 256): w = t·1000·exp(-ln(10000)·i/128), [sin w, cos w]."""
        half = self.cfg.time_embedding_dim // 2
        f = torch.exp(-math.log(_TIME_BASE) * torch.arange(half, device=t.device) / half)
        w = t[:, None] * self.cfg.time_scale * f[None]
        return torch.cat([w.sin(), w.cos()], -1)

    def condition(self, t: torch.Tensor, dit_vector: torch.Tensor) -> torch.Tensor:
        """Per-sequence condition c = time_mlp(time_embedding(t)) + projected voice vector, (B, D)."""
        return self.time_mlp(self.time_embedding(t).to(dit_vector.dtype)) + dit_vector

    def _rope(self, length: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        return rope_table(length, self.cfg.dim // self.heads, self.cfg.rope_base, device)

    @torch.no_grad()
    def prompt_cache(self, prompt: torch.Tensor, dit_tokens: torch.Tensor, dit_vector: torch.Tensor) -> list[KV]:
        """Run the clean prompt frames (B, P, 96) once at t = 1 and keep every block's keys and values."""
        b, p, _ = prompt.shape
        one = torch.ones(b, p, 1, dtype=prompt.dtype, device=prompt.device)
        x = self.input_proj(torch.cat([torch.zeros_like(prompt), prompt, one], -1))
        c = self.condition(torch.ones(b, device=prompt.device), dit_vector)
        rope = self._rope(p, x.device)
        cache: list[KV] = []
        for blk in self.blocks:
            x = blk(x, c, rope, blk.cross_attn.keys_values(dit_tokens), keep=cache)
        return cache

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor, frame_cond: torch.Tensor, voice: VoiceState) -> torch.Tensor:
        """Velocity for the new frames only. x_t (B, L, 96); t (B,); frame_cond (B, L, 512) text features of
        these frames; the prompt sits in front of them through ``voice.prompt_kv``."""
        b, l, _ = x_t.shape
        p = voice.prompt_frames
        zero = torch.zeros_like(x_t)
        x = self.input_proj(torch.cat([x_t, zero, zero[..., :1]], -1)) + self.text_proj(frame_cond)
        c = self.condition(t, voice.dit_vector)
        cos, sin = self._rope(p + l, x.device)
        rope = (cos[p:], sin[p:])
        for blk, cross_kv, prompt_kv in zip(self.blocks, voice.cross_kv, voice.prompt_kv):
            x = blk(x, c, rope, cross_kv, prompt_kv)
        return self.output_proj(self.final_norm(x, self.final_norm.modulation(c)))


# --------------------------------------------------------------------------------------------- the whole model
class AcousticModel(nn.Module):
    """Text encoder, voice encoder, duration predictor and DiT; weights in ``model.safetensors``."""

    def __init__(self, cfg: ArticulinoConfig):
        super().__init__()
        self.cfg = cfg
        self.text_encoder = TextEncoder(cfg.text.vocab_size, cfg.text_encoder)
        self.voice_encoder = VoiceEncoder(cfg.voice_encoder)
        self.duration_predictor = DurationPredictor(cfg.duration_predictor)
        self.dit = DiT(cfg.dit)

    @torch.no_grad()
    def prepare_voice(self, latents: torch.Tensor, prompt_frames: int | None = None) -> VoiceState:
        """latents (1, R, 96): normalized reference latents. Computes the voice tokens/vector and the voice cache."""
        p = self.cfg.sampler.prompt_frames if prompt_frames is None else prompt_frames
        tokens, vector = self.voice_encoder(latents)
        dit_tokens = self.dit.voice_token_proj(tokens)
        dit_vector = self.dit.voice_vector_proj(vector)
        prompt = latents[:, :p]
        return VoiceState(
            latents=latents, tokens=tokens, vector=vector, prompt=prompt, dit_vector=dit_vector,
            cross_kv=[blk.cross_attn.keys_values(dit_tokens) for blk in self.dit.blocks],
            prompt_kv=self.dit.prompt_cache(prompt, dit_tokens, dit_vector),
        )
