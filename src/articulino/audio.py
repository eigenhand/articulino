"""Reading reference recordings and resampling them to 24 kHz."""
from __future__ import annotations

import math
import os

import numpy as np
import torch


def load_audio(path: str | os.PathLike) -> tuple[np.ndarray, int]:
    """Read an audio file as mono float32 (channels averaged). Returns (samples, sample_rate)."""
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(1)
    return np.ascontiguousarray(wav, dtype=np.float32), int(sr)


def resample(wav: np.ndarray, orig_sr: int, new_sr: int, lowpass_filter_width: int = 6,
             rolloff: float = 0.99) -> np.ndarray:
    """Band-limited (Hann-windowed sinc) resampling of mono audio.

    Same algorithm and defaults as ``torchaudio.functional.resample`` (``sinc_interp_hann``), so the package does
    not need torchaudio."""
    if orig_sr == new_sr:
        return np.asarray(wav, dtype=np.float32)
    gcd = math.gcd(int(orig_sr), int(new_sr))
    orig, new = int(orig_sr) // gcd, int(new_sr) // gcd
    x = torch.as_tensor(np.asarray(wav, dtype=np.float32))[None]
    base = min(orig, new) * rolloff
    width = math.ceil(lowpass_filter_width * orig / base)
    idx = torch.arange(-width, width + orig, dtype=torch.float32)[None, None] / orig
    t = torch.arange(0, -new, -1, dtype=torch.float32)[:, None, None] / new + idx
    t *= base
    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)
    window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    t *= math.pi
    scale = base / orig
    kernels = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernels = kernels * window * scale
    length = x.shape[-1]
    x = torch.nn.functional.pad(x, (width, width + orig))
    y = torch.nn.functional.conv1d(x[:, None], kernels, stride=orig)
    y = y.transpose(1, 2).reshape(1, -1)
    target = math.ceil(new * length / orig)
    return y[0, :target].numpy().astype(np.float32)
