"""Durations and the 4-step sampler of the distilled student (DMD2 style, no classifier-free guidance).

From pure noise, each step predicts the clean latents ``x1 = x + (1 - t_k)·v(x, t_k)``; the next step starts from
those latents re-noised to the next time, ``x = (1 - t_{k+1})·noise_{k+1} + t_{k+1}·x1``. The result is ``x1`` of
the last step. The times are fixed by training: t = 0, 0.25, 0.5, 0.75.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .model import DiT, VoiceState
from .text import CHARSET

_IS_LETTER = np.array([False] + [c.isalnum() for c in CHARSET])


def letter_rate(durations: torch.Tensor, ids: torch.Tensor, start: int = 0,
                target: float | None = None) -> tuple[list[float], torch.Tensor]:
    """Speaking rate in letters per second, counted over letters and digits from position ``start`` (pauses at
    spaces and punctuation do not count). Single over-long letters (silence the aligner gave to one letter) are
    capped at three times the median.

    With ``target``, the letter durations are rescaled so the rate becomes exactly ``target`` (returned as the second
    value; otherwise the durations come back unchanged). durations (B, N) in 10 ms units, ids (B, N)."""
    d_all = durations.detach().float().cpu().numpy().copy()
    ids_np = ids.cpu().numpy()
    rates = []
    for i in range(ids_np.shape[0]):
        m = _IS_LETTER[ids_np[i]].copy()
        m[:start] = False
        if not m.any():
            rates.append(0.0)
            continue
        d = np.maximum(d_all[i, m], 0.5)
        d = np.minimum(d, 3 * np.median(d))
        rate = float(m.sum() / (d.sum() / 100))
        if target:
            d = d * (rate / target)
            rate = target
        d_all[i, m] = d
        rates.append(rate)
    if target is None:
        return rates, durations
    return rates, torch.from_numpy(d_all).to(durations.device, durations.dtype)


def draw_noise(seed: int, total_frames: int, prompt_frames: int, steps: int,
               latent_dim: int = 96) -> list[torch.Tensor]:
    """The noise for every step, (1, total_frames - prompt_frames, latent_dim) each, drawn on the CPU.

    Drawn over all frames (prompt included) and then cut, exactly like the original sampler, so a seed gives the
    same noise as there; drawn on the CPU, so it is the same on every device."""
    gen = torch.Generator().manual_seed(int(seed))
    return [torch.randn(1, total_frames, latent_dim, generator=gen)[:, prompt_frames:] for _ in range(steps)]


@torch.no_grad()
def sample(dit: DiT, frame_cond: torch.Tensor, voice: VoiceState, timesteps: Sequence[float],
           noise: Sequence[torch.Tensor]) -> torch.Tensor:
    """Generate the latents of the new frames.

    frame_cond (B, L, 512): text features per new frame; noise: one tensor (B, L, 96) per step (from
    :func:`draw_noise`, or given from outside for tests). Returns the clean normalized latents (B, L, 96)."""
    if len(noise) < len(timesteps):
        raise ValueError(f"need {len(timesteps)} noise tensors, got {len(noise)}")
    b = frame_cond.shape[0]
    x = noise[0].to(frame_cond.device, frame_cond.dtype)
    x1 = x
    for k, t in enumerate(timesteps):
        v = dit.velocity(x, torch.full((b,), float(t), device=x.device), frame_cond, voice).to(x.dtype)
        x1 = x + (1 - t) * v
        if k + 1 < len(timesteps):
            t_next = timesteps[k + 1]
            x = (1 - t_next) * noise[k + 1].to(x.device, x.dtype) + t_next * x1
    return x1
