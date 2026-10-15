"""Smoke tests of the public API: say(), stream(), set_voice(), the CLI."""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import ROOT, VECTORS

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def voice_wav():
    path = VECTORS / "de_sentence.npz"
    if not path.is_file():
        pytest.skip("reference voice (from the vectors) missing")
    return np.load(path)["voice_wav"]


@pytest.fixture(scope="module")
def speaker(tts, voice_wav):
    tts.set_voice(voice_wav)
    return tts


def test_say(speaker, tmp_path):
    import soundfile as sf

    from articulino import Speech

    out = tmp_path / "morgen.wav"
    speech = speaker.say("Guten Morgen! Der Termin ist um 15:30 Uhr.", path=out)
    assert isinstance(speech, Speech)
    assert speech.sample_rate == 24000 and speech.seed == 0
    assert speech.audio.dtype == np.float32 and speech.audio.ndim == 1
    assert abs(speech.duration - len(speech.audio) / 24000) < 1e-9
    assert 1.5 < speech.duration < 6.0
    assert np.abs(speech.audio).max() <= 0.95 + 1e-6
    data, sr = sf.read(out, dtype="float32")
    assert sr == 24000 and np.abs(data - speech.audio).max() <= 1 / 32768


def test_seed_is_deterministic(speaker):
    a = speaker.say("Das ist ein Satz zum Vergleichen von zwei Läufen.", seed=3).audio
    b = speaker.say("Das ist ein Satz zum Vergleichen von zwei Läufen.", seed=3).audio
    c = speaker.say("Das ist ein Satz zum Vergleichen von zwei Läufen.", seed=4).audio
    assert np.array_equal(a, b)
    assert len(a) != len(c) or not np.allclose(a, c)


def test_speed(speaker):
    text = "Draußen sind es zwölf Grad, und am Abend soll es regnen."
    slow, fast = speaker.say(text, speed=0.8).duration, speaker.say(text, speed=1.4).duration
    assert slow > fast * 1.4
    with pytest.raises(ValueError):
        speaker.say(text, speed=0)


def test_stream_matches_say(speaker):
    """The card's example: stream() yields one chunk per sentence, the same audio as say() without the pauses."""
    pieces = ["Guten Mor", "gen! Heute ", "regnet es ab 18 Uhr. ", "Nimm einen Schirm mit."]
    chunks = list(speaker.stream(iter(pieces)))
    assert len(chunks) == 2 and all(c.dtype == np.float32 for c in chunks)
    whole = speaker.say("".join(pieces)).audio
    gap = np.zeros(int(0.15 * 24000), dtype=np.float32)
    assert np.array_equal(np.concatenate([chunks[0], gap, chunks[1]]), whole)
    assert len(list(speaker.stream("Ein ganzer Text als Zeichenkette, nicht als Liste."))) == 1


def test_language_and_tags(speaker):
    en = speaker.say("The delivery arrives tomorrow between ten and twelve.", language="en")
    assert en.duration > 1.5
    mixed = speaker.say("Das [en]Meeting[/en] ist auf morgen verschoben.")
    assert mixed.duration > 1.0
    with pytest.raises(ValueError):
        speaker.say("Bonjour", language="fr")


def test_options(tts, voice_wav):
    from articulino import TTS

    calls = []
    plain = TTS(tts.model, tts.codec, tts.config, "cpu", uniform_rate=False, low_shelf_db=0, peak_limit=None,
                watermark=lambda a, sr: calls.append(sr) or a)
    voice = plain.set_voice(voice_wav)
    assert voice.rate is None and abs(voice.seconds - len(voice_wav) / 24000) < 1e-9
    s = plain.say("Ein Satz ohne Nachbearbeitung, aber mit Haken.")
    assert s.duration > 1 and calls == [24000]


def test_set_voice_variants(tts, voice_wav, tmp_path):
    import soundfile as sf

    from articulino import TTS
    from articulino.audio import resample

    t2 = TTS(tts.model, tts.codec, tts.config, "cpu")
    path = tmp_path / "v48.wav"
    sf.write(path, np.stack([resample(voice_wav, 24000, 48000)] * 2, 1), 48000)    # stereo, 48 kHz
    v = t2.set_voice(path)
    assert 4.5 < v.seconds <= 8.0 and v.rate and 8 < v.rate < 25
    assert t2.set_voice(v) is v
    with pytest.raises(ValueError):
        t2.set_voice(np.zeros(1000, dtype=np.float32))
    with pytest.raises(RuntimeError):
        TTS(tts.model, tts.codec, tts.config, "cpu").say("Ohne Stimme.")


def test_cli(model_dir, voice_wav, tmp_path):
    import soundfile as sf

    voice = tmp_path / "voice.wav"
    sf.write(voice, voice_wav, 24000)
    out = tmp_path / "cli.wav"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-m", "articulino", "say", "Hallo, das ist ein Test der Kommandozeile.",
                        "--voice", str(voice), "-o", str(out), "--model", str(model_dir), "--device", "cpu",
                        "--threads", "4"], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr
    audio, sr = sf.read(out)
    assert sr == 24000 and len(audio) > 24000
    r = subprocess.run([sys.executable, "-m", "articulino", "normalize", "Am 3.10. um 9:05 Uhr."], capture_output=True,
                       text=True, env=env, timeout=120)
    assert r.stdout.strip() == "Am dritten Oktober um neun Uhr fünf."


def test_from_pretrained_repo_id(model_dir, monkeypatch):
    """A repository id goes through huggingface_hub.snapshot_download (mocked: no network, nothing uploaded)."""
    hub = pytest.importorskip("huggingface_hub")
    from articulino import DEFAULT_REPO, TTS

    calls = []

    def fake_snapshot_download(repo_id, **kw):
        calls.append((repo_id, kw))
        return str(model_dir)

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)
    tts = TTS.from_pretrained(DEFAULT_REPO, device="cpu", revision="main")
    assert calls and calls[0][0] == DEFAULT_REPO and calls[0][1]["revision"] == "main"
    assert set(calls[0][1]["allow_patterns"]) == {"config.json", "model.safetensors", "codec.safetensors"}
    assert tts.device.type == "cpu"
    with pytest.raises(FileNotFoundError):
        TTS.from_pretrained(model_dir.parent)          # a folder without the weights
