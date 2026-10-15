"""Resampling of reference recordings."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from articulino.audio import resample


@pytest.mark.parametrize("sr", [48000, 44100, 22050, 16000, 8000])
def test_resample_like_torchaudio(sr):
    ta = pytest.importorskip("torchaudio.functional")
    x = np.random.default_rng(sr).standard_normal(sr * 2).astype(np.float32) * 0.2
    a = resample(x, sr, 24000)
    b = ta.resample(torch.from_numpy(x), sr, 24000).numpy()
    assert a.shape == b.shape == (48000,)
    assert np.abs(a - b).max() < 1e-6


def test_resample_identity_and_tone():
    x = np.sin(2 * np.pi * 440 * np.arange(48000) / 48000).astype(np.float32)
    assert resample(x, 24000, 24000) is not None and np.array_equal(resample(x, 24000, 24000), x)
    y = resample(x, 48000, 24000)
    ref = np.sin(2 * np.pi * 440 * np.arange(24000) / 24000)
    assert np.abs(y[100:-100] - ref[100:-100]).max() < 1e-2
