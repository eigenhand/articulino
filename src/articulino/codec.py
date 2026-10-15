"""Latent codec: 24 kHz audio <-> 96-dim latents at 25 Hz (weights in ``codec.safetensors``).

* :class:`LogMel`: the encoder's input features, identical to torchaudio's ``MelSpectrogram`` (n_fft 1024,
  win 960, hop 240, 100 HTK mel bands without filter normalization, magnitude, reflect padding) followed by
  ``log(clamp(x, 1e-5))``. Implemented here so the package does not need torchaudio.
* :class:`LatentEncoder`: log-mel (100 Hz) -> ConvNeXt stages with two stride-2 convolutions -> mean latents (25 Hz).
  Only used once per voice.
* :class:`LatentDecoder`: latents (25 Hz) -> 4x transposed convolution (100 Hz) -> 12 ConvNeXt blocks -> log
  magnitude and phase per STFT bin -> inverse STFT (n_fft 960, hop 240) -> 24 kHz audio. Vocos style: no upsampling
  in the time domain, which keeps it fast on a CPU.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 24000
HOP = 240                 # 100 Hz feature frames
N_FFT = 960               # inverse STFT of the decoder
MEL_N_FFT = 1024
MEL_WIN = 960
N_MELS = 100
LATENT_HOP = 960          # samples per latent frame (25 Hz)


def mel_filterbank(n_freqs: int = MEL_N_FFT // 2 + 1, f_min: float = 0.0, f_max: float = SAMPLE_RATE / 2,
                   n_mels: int = N_MELS, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Triangular HTK mel filters without normalization, (n_freqs, n_mels); same numbers as
    ``torchaudio.functional.melscale_fbanks(..., norm=None, mel_scale="htk")``."""
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
    m_min = 2595.0 * math.log10(1.0 + f_min / 700.0)
    m_max = 2595.0 * math.log10(1.0 + f_max / 700.0)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = 700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    down = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    return torch.max(torch.zeros(1), torch.min(down, up))


class LogMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(MEL_WIN), persistent=False)      # periodic
        self.register_buffer("filters", mel_filterbank(), persistent=False)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav (B, T) -> log-mel (B, 100, T // 240 + 1)."""
        spec = torch.stft(wav, MEL_N_FFT, HOP, MEL_WIN, window=self.window, center=True, pad_mode="reflect",
                          normalized=False, onesided=True, return_complex=True).abs()
        mel = torch.matmul(spec.transpose(-1, -2), self.filters).transpose(-1, -2)
        return torch.log(mel.clamp(min=1e-5))


class ConvNeXtBlock(nn.Module):
    """Depthwise conv (k 7) -> LayerNorm (eps 1e-6) -> Linear ×3 -> GELU -> Linear -> x + scale·y."""

    def __init__(self, dim: int, inner: int):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.up = nn.Linear(dim, inner)
        self.down = nn.Linear(inner, dim)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x (B, C, T)
        y = self.depthwise_conv(x).transpose(1, 2)
        y = self.down(F.gelu(self.up(self.norm(y))))
        return x + (self.scale * y).transpose(1, 2)


class LatentEncoder(nn.Module):
    def __init__(self, dim: int = 512, latent_dim: int = 96, blocks_per_stage: int = 4):
        super().__init__()
        self.blocks_per_stage = blocks_per_stage
        self.mel = LogMel()
        self.input_conv = nn.Conv1d(N_MELS, dim, 7, padding=3)
        self.blocks = nn.ModuleList(ConvNeXtBlock(dim, 3 * dim) for _ in range(3 * blocks_per_stage))
        self.downsample = nn.ModuleList(nn.Conv1d(dim, dim, 4, stride=2, padding=1) for _ in range(2))
        self.final_norm = nn.LayerNorm(dim, eps=1e-6)
        self.output_proj = nn.Linear(dim, latent_dim)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav (B, T) with T a multiple of 960 -> mean latents (B, 96, T // 960)."""
        x = self.input_conv(self.mel(wav)[..., : wav.shape[-1] // HOP])
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i % self.blocks_per_stage == self.blocks_per_stage - 1 and i // self.blocks_per_stage < 2:
                x = self.downsample[i // self.blocks_per_stage](x)
        return self.output_proj(self.final_norm(x.transpose(1, 2))).transpose(1, 2)


class LatentDecoder(nn.Module):
    def __init__(self, latent_dim: int = 96, dim: int = 512, blocks: int = 12, max_magnitude: float = 100.0):
        super().__init__()
        self.max_magnitude = max_magnitude
        self.input_conv = nn.Conv1d(latent_dim, dim, 3, padding=1)
        self.upsample = nn.ConvTranspose1d(dim, dim, 8, stride=4, padding=2)          # 25 -> 100 Hz
        self.blocks = nn.ModuleList(ConvNeXtBlock(dim, 3 * dim) for _ in range(blocks))
        self.final_norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, N_FFT + 2)                                          # 481 log-magnitudes + 481 phases
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z (B, 96, L) de-normalized latents -> audio (B, L·960)."""
        h = self.upsample(self.input_conv(z))
        for block in self.blocks:
            h = block(h)
        h = self.head(self.final_norm(h.transpose(1, 2))).transpose(1, 2)
        log_mag, phase = h.chunk(2, dim=1)
        mag = torch.exp(log_mag).clamp(max=self.max_magnitude)
        spec = torch.polar(mag.float(), phase.float())
        return torch.istft(spec, N_FFT, HOP, N_FFT, window=self.window.float(), center=True,
                           length=z.shape[-1] * LATENT_HOP)


class Codec(nn.Module):
    def __init__(self, latent_dim: int = 96, encoder_dim: int = 512, decoder_dim: int = 512, decoder_blocks: int = 12,
                 blocks_per_stage: int = 4, max_magnitude: float = 100.0):
        super().__init__()
        self.encoder = LatentEncoder(encoder_dim, latent_dim, blocks_per_stage)
        self.decoder = LatentDecoder(latent_dim, decoder_dim, decoder_blocks, max_magnitude)

    @classmethod
    def from_config(cls, codec_cfg: dict, latent_dim: int = 96) -> "Codec":
        enc, dec = codec_cfg.get("encoder", {}), codec_cfg.get("decoder", {})
        return cls(latent_dim=latent_dim, encoder_dim=enc.get("dim", 512), decoder_dim=dec.get("dim", 512),
                   decoder_blocks=dec.get("blocks", 12), blocks_per_stage=enc.get("blocks_per_stage", 4),
                   max_magnitude=dec.get("max_magnitude", 100.0))
