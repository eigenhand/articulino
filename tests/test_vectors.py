"""Every building block against the reference vectors computed with the original inference code (CPU, float32).

Tolerance (docs/SPEC.md): max|diff| / max|ref| < 1e-3 per block; audio SNR > 40 dB. The noise of the four sampling
steps comes from the vectors. Each block gets the *reference* inputs, so errors do not carry over between blocks.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from conftest import VECTORS, record

TOL = 1e-3
CASES = ["de_sentence", "en_sentence"]


def rel(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    return float(np.abs(a - b).max() / np.abs(b).max())


@pytest.fixture(scope="module", params=CASES)
def case(request):
    path = VECTORS / f"{request.param}.npz"
    if not path.is_file():
        pytest.skip(f"reference vectors {path} missing")
    v = dict(np.load(path))
    info = json.loads((VECTORS / f"{request.param}.json").read_text(encoding="utf-8"))
    return request.param, v, info


def t(v, key):
    return torch.from_numpy(v[key])[None]


def check(block, name, value, tol=TOL):
    record(block, name, value)
    assert value < tol, f"{block}: {value:.3g} >= {tol}"


@torch.inference_mode()
def test_log_mel(raw_tts, case):
    name, v, _ = case
    check("log-mel", name, rel(raw_tts.codec.encoder.mel(t(v, "voice_wav"))[0], v["voice_logmel"]))


@torch.inference_mode()
def test_latent_encoder(raw_tts, case):
    name, v, _ = case
    mu = raw_tts.codec.encoder(t(v, "voice_wav"))
    check("encoder mu", name, rel(mu[0], v["voice_mu"]))
    ref = (t(v, "voice_mu").transpose(1, 2) - raw_tts._mean) / raw_tts._std
    check("normalized latents", name, rel(ref[0], v["voice_ref"]))


@torch.inference_mode()
def test_text_encoder(raw_tts, case):
    name, v, info = case
    lang = torch.tensor([raw_tts.config.text.languages[info["language"]]])
    h = raw_tts.model.text_encoder(t(v, "text_ids"), lang, t(v, "text_mask"))
    check("text encoder", name, rel(h[0], v["text_h"]))


def test_text_ids(case):
    from articulino.text import encode

    name, v, info = case
    ids, langs = encode(info["sentence"], info["language"])
    assert ids == v["text_ids_raw"].tolist()
    assert v["text_ids"].tolist() == [0] + ids and v["text_mask"].tolist() == [False] + [True] * len(ids)


@torch.inference_mode()
def test_voice_encoder(raw_tts, case):
    name, v, _ = case
    tokens, vector = raw_tts.model.voice_encoder(t(v, "voice_ref"))
    check("voice tokens", name, rel(tokens[0], v["voice_tokens"]))
    check("voice vector", name, rel(vector[0], v["voice_vector"]))


@torch.inference_mode()
def test_durations(raw_tts, case):
    from articulino.model import expand_to_frames

    name, v, info = case
    log_d = raw_tts.model.duration_predictor(t(v, "text_h"), t(v, "voice_vector"), t(v, "text_mask"))
    check("duration (log)", name, rel(log_d[0], v["duration_log"]))
    d = (t(v, "duration_log").exp() * t(v, "text_mask")).clamp(max=400)
    d[:, 0] = 4 * info["prompt_frames"]
    check("durations", name, rel(d[0], v["durations"]))
    frames = int(torch.ceil(d.sum(1) / 4).item())
    assert frames == info["frames_total"]
    check("frame condition", name, rel(expand_to_frames(t(v, "text_h"), t(v, "durations"), frames)[0], v["frame_cond"]))


@torch.inference_mode()
def test_voice_cache(raw_tts, case):
    name, v, info = case
    state = raw_tts.model.prepare_voice(t(v, "voice_ref"), info["prompt_frames"])
    err = max(max(rel(k[0], v[f"cache_k{i}"]), rel(val[0], v[f"cache_v{i}"])) for i, (k, val) in enumerate(state.prompt_kv))
    check("voice cache (k, v)", name, err)


def _reference_state(model, v, prompt_frames):
    """Voice state built from the reference tokens, vector and cache (isolates the DiT)."""
    from articulino.model import VoiceState

    tokens, vector = t(v, "voice_tokens"), t(v, "voice_vector")
    dit_tokens = model.dit.voice_token_proj(tokens)
    return VoiceState(latents=t(v, "voice_ref"), tokens=tokens, vector=vector, prompt=t(v, "voice_ref")[:, :prompt_frames],
                      dit_vector=model.dit.voice_vector_proj(vector),
                      cross_kv=[b.cross_attn.keys_values(dit_tokens) for b in model.dit.blocks],
                      prompt_kv=[(t(v, f"cache_k{i}"), t(v, f"cache_v{i}")) for i in range(len(model.dit.blocks))])


@torch.inference_mode()
def test_dit_velocity(raw_tts, case):
    name, v, info = case
    p = info["prompt_frames"]
    state = _reference_state(raw_tts.model, v, p)
    cond = t(v, "frame_cond")[:, p:]
    for k, tk in enumerate(info["timesteps"]):
        vel = raw_tts.model.dit.velocity(t(v, f"step{k}_in"), torch.full((1,), float(tk)), cond, state)
        check(f"DiT velocity step {k}", name, rel(vel[0], v[f"step{k}_velocity"]))


@torch.inference_mode()
def test_sampler(raw_tts, case):
    from articulino.sampler import sample

    name, v, info = case
    p = info["prompt_frames"]
    state = _reference_state(raw_tts.model, v, p)
    x1 = sample(raw_tts.model.dit, t(v, "frame_cond")[:, p:], state, info["timesteps"], [t(v, f"noise_{k}") for k in range(4)])
    check("4-step latents", name, rel(x1[0], v["step3_x1"]))


@torch.inference_mode()
def test_decoder(raw_tts, case):
    name, v, _ = case
    check("decoder (iSTFT)", name, rel(raw_tts.codec.decoder(t(v, "decoder_in"))[0], v["decoder_out"]))


def test_end_to_end(raw_tts, case):
    """Voice wav + sentence + given noise -> audio, through the public pipeline (no uniform rate, no post-processing)."""
    name, v, info = case
    raw_tts.set_voice(v["voice_wav"])
    lang = raw_tts.config.text.languages[info["language"]]
    raw, skip = raw_tts.synthesize(v["text_ids_raw"].tolist(), lang, noise=[t(v, f"noise_{k}") for k in range(4)])
    audio, ref = raw[skip:], v["audio"]
    assert audio.shape == ref.shape
    snr = 10 * np.log10((ref.astype(np.float64) ** 2).sum() / ((audio - ref).astype(np.float64) ** 2).sum())
    record("audio SNR (dB)", name, snr)
    assert snr > 40
