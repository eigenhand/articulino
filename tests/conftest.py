"""Shared fixtures.

Environment variables:
  ARTICULINO_MODEL_DIR   folder with config.json, model.safetensors, codec.safetensors (default: build/release)
  ARTICULINO_VECTORS     folder with the reference vectors (default: tests/vectors)
  ARTICULINO_ASR_URL     speech recognizer for the listening check, e.g. http://127.0.0.1:8084 (opt-in)
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

MODEL_DIR = Path(os.environ.get("ARTICULINO_MODEL_DIR", ROOT / "build" / "release"))
VECTORS = Path(os.environ.get("ARTICULINO_VECTORS", HERE / "vectors"))
REFERENCE_VOICE = HERE / "data" / "radiov.wav"

# block -> {vector name: relative error}; printed at the end of the run
RESULTS: dict[str, dict[str, float]] = defaultdict(dict)


def record(block: str, case: str, value: float) -> None:
    RESULTS[block][case] = value


def pytest_terminal_summary(terminalreporter):
    if not RESULTS:
        return
    cases = sorted({c for v in RESULTS.values() for c in v})
    terminalreporter.write_sep("-", "equality against the reference vectors (max|diff| / max|ref|; audio: SNR dB)")
    terminalreporter.write_line(f"{'block':<22}" + "".join(f"{c:>14}" for c in cases))
    for block, vals in RESULTS.items():
        cells = "".join(f"{vals[c]:>14.3g}" if c in vals else f"{'-':>14}" for c in cases)
        terminalreporter.write_line(f"{block:<22}{cells}")


@pytest.fixture(scope="session")
def model_dir() -> Path:
    if not (MODEL_DIR / "model.safetensors").is_file():
        pytest.skip(f"no exported model in {MODEL_DIR} (run tools/export_safetensors.py)")
    return MODEL_DIR


@pytest.fixture(scope="session")
def tts(model_dir):
    """The model as released (default options), on the CPU."""
    import torch

    from articulino import TTS

    torch.set_num_threads(min(8, torch.get_num_threads()))
    return TTS.from_pretrained(model_dir, device="cpu")


@pytest.fixture(scope="session")
def raw_tts(tts):
    """Same weights, but without the uniform rate and without post-processing (as in the reference vectors)."""
    from articulino import TTS

    return TTS(tts.model, tts.codec, tts.config, "cpu", uniform_rate=False, low_shelf_db=0, peak_limit=None)


@pytest.fixture(scope="session")
def voice_path() -> Path:
    if not REFERENCE_VOICE.is_file():
        pytest.skip(f"reference voice {REFERENCE_VOICE} missing")
    return REFERENCE_VOICE
