"""The released files: readable without the safetensors library, complete, and consistent with config.json."""
from __future__ import annotations

import json

import pytest
import torch

from articulino._safetensors import load_file
from articulino.codec import Codec
from articulino.config import ArticulinoConfig
from articulino.model import AcousticModel


def test_reader_matches_safetensors_library(model_dir):
    st = pytest.importorskip("safetensors.torch")
    for name in ("model.safetensors", "codec.safetensors"):
        mine, meta = load_file(model_dir / name)
        ref = st.load_file(str(model_dir / name))
        assert mine.keys() == ref.keys() and meta.get("model_type") == "articulino"
        assert all(torch.equal(mine[k], ref[k]) for k in ref)


def test_files_match_the_modules(model_dir):
    cfg = ArticulinoConfig.from_file(model_dir / "config.json")
    model, codec = AcousticModel(cfg), Codec.from_config(cfg.codec, cfg.latents.dim)
    for module, name in ((model, "model.safetensors"), (codec, "codec.safetensors")):
        tensors, _ = load_file(model_dir / name)
        expected = module.state_dict()
        assert tensors.keys() == expected.keys()
        assert all(tensors[k].shape == expected[k].shape and tensors[k].dtype == torch.float32 for k in tensors)
    n_model = sum(t.numel() for t in load_file(model_dir / "model.safetensors")[0].values())
    n_codec = sum(t.numel() for t in load_file(model_dir / "codec.safetensors")[0].values())
    assert round(n_model / 1e6, 1) == 66.3 and round(n_codec / 1e6, 1) == 43.2


def test_config(model_dir):
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    cfg = ArticulinoConfig.from_dict(raw)
    assert cfg.sample_rate == 24000 and cfg.text.vocab_size == len(cfg.text.charset) + 1 == 53
    assert len(cfg.latents.mean) == len(cfg.latents.std) == cfg.latents.dim == 96
    assert min(cfg.latents.std) > 0
    assert cfg.sampler.timesteps == (0.0, 0.25, 0.5, 0.75) and cfg.sampler.prompt_frames == 125
    from articulino.text import CHARSET

    assert cfg.text.charset == CHARSET
    with pytest.raises(ValueError):
        ArticulinoConfig.from_dict({"model_type": "something-else"})
