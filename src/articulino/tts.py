"""The high-level API: :class:`TTS` and :class:`Speech`."""
from __future__ import annotations

import math
import os
import warnings
from collections.abc import Callable, Iterable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import frontend
from ._safetensors import load_file
from .audio import load_audio, resample
from .codec import Codec
from .config import ArticulinoConfig
from .model import AcousticModel, VoiceState, expand_to_frames
from .postprocess import SosFilter, limit_peak, shelf_sos
from .sampler import draw_noise, letter_rate, sample
from .sentences import SentenceBuffer, carry_language_tags, split_sentences
from .text import encode, open_language_tag, protect_foreign_spans, restore_foreign_spans

DEFAULT_REPO = "eigenhand/Articulino-DE-EN-Realtime-Stream-TTS"
WEIGHT_FILES = ("config.json", "model.safetensors", "codec.safetensors")
SENTENCE_PAUSE = 0.15               # seconds of silence between sentences in say()
MIN_SPEED, MAX_SPEED = 0.7, 1.6
MIN_VOICE_SECONDS = 1.0

# Neutral German sentences used once per voice to measure the voice's natural speaking rate (uniform_rate=True).
RATE_CALIBRATION = (
    "Guten Morgen, die Systeme laufen stabil und es gibt nichts Dringendes.",
    "Ich habe die Unterlagen geprüft und fasse das Wichtigste kurz zusammen.",
    "Der Termin am Donnerstag wurde auf den Nachmittag verschoben.",
    "Soll ich die Nachricht jetzt abschicken oder noch einmal vorlesen?",
    "Das Training läuft seit gestern Abend ohne Unterbrechung weiter.",
    "Draußen sind es zwölf Grad, und am Abend soll es regnen.",
)

Watermark = Callable[[np.ndarray, int], np.ndarray]


def _normalize_german(text: str) -> str:
    """The German frontend, except for spans tagged [en]…[/en]: those keep their spelling (only the model's own
    character normalization applies), so English words are not read with German letter names or numbers."""
    masked, spans = protect_foreign_spans(text, "de")
    if not spans:
        return frontend.normalize(text)
    restored = restore_foreign_spans(frontend.normalize(masked), spans)
    return frontend.normalize(text) if restored is None else restored


class _TaggedStreamNormalizer(frontend.StreamNormalizer):
    """StreamNormalizer with [en]…[/en] protection; a tag left open in one released chunk continues in the next."""

    def __init__(self) -> None:
        super().__init__(self._normalize_chunk)
        self._open: str | None = None

    def _normalize_chunk(self, chunk: str) -> str:
        prefix = self._open or ""
        out = _normalize_german(prefix + chunk)
        self._open = open_language_tag(prefix + chunk)
        return out[len(prefix):] if prefix and out.startswith(prefix) else out


class _FromConfig:
    """Default marker: take the value from config.json."""

    def __repr__(self) -> str:
        return "FROM_CONFIG"


FROM_CONFIG = _FromConfig()


@dataclass(frozen=True)
class Speech:
    """Generated audio: float32 mono samples in [-1, 1], their sample rate, length in seconds and the seed."""

    audio: np.ndarray
    sample_rate: int
    duration: float
    seed: int

    def save(self, path: str | os.PathLike, subtype: str = "PCM_16") -> None:
        """Write the audio to a file (format from the extension, e.g. .wav or .flac; 16-bit PCM by default)."""
        import soundfile as sf

        sf.write(str(path), self.audio, self.sample_rate, subtype=subtype)


@dataclass
class Voice:
    """A prepared reference voice (returned by :meth:`TTS.set_voice`, can be passed back to it later)."""

    state: VoiceState
    rate: float | None          # natural speaking rate in letters per second (None if uniform_rate is off)
    seconds: float              # length of the reference that was used


class TTS:
    """German and English text to speech with zero-shot voice cloning.

    Create it with :meth:`from_pretrained`, choose a voice with :meth:`set_voice`, then call :meth:`say` for whole
    texts or :meth:`stream` for text that arrives piece by piece.
    """

    def __init__(self, model: AcousticModel, codec: Codec, config: ArticulinoConfig, device: str | torch.device = "cpu",
                 *, uniform_rate: bool = True, low_shelf_db: float | None = FROM_CONFIG,
                 peak_limit: float | None = FROM_CONFIG, watermark: Watermark | None = None):
        """Prefer :meth:`from_pretrained`. Options:

        uniform_rate: bring every sentence to the voice's natural speaking rate (× ``speed``), measured once per voice.
            Keeps the tempo steady from sentence to sentence. With False, the duration predictor's rate is used as is.
        low_shelf_db: gain of the low-shelf at 180 Hz applied after the decoder; default from the config (+2 dB);
            0 or None switches it off.
        peak_limit: scale a sentence down if its peak is above this; default from the config (0.95); None switches
            it off.
        watermark: reserved hook, ``f(audio, sample_rate) -> audio``, applied to every chunk after post-processing.
            Articulino ships no watermark; see the model card on marking synthetic audio.
        """
        self.config = config
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.codec = codec.to(self.device).eval()
        self.sample_rate = config.sample_rate
        self.uniform_rate = uniform_rate
        pp = config.postprocess
        self.low_shelf_db = float(pp.low_shelf_db if low_shelf_db is FROM_CONFIG else (low_shelf_db or 0.0))
        self.peak_limit = pp.peak_limit if peak_limit is FROM_CONFIG else peak_limit
        self.watermark = watermark
        self._shelf = (SosFilter(shelf_sos(pp.low_shelf_hz, self.low_shelf_db, False, self.sample_rate,
                                           pp.shelf_slope)) if self.low_shelf_db else None)
        self._mean = torch.tensor(config.latents.mean, dtype=torch.float32, device=self.device)
        self._std = torch.tensor(config.latents.std, dtype=torch.float32, device=self.device)
        self.voice: Voice | None = None

    # ------------------------------------------------------------------------------------------- loading
    @classmethod
    def from_pretrained(cls, repo_or_dir: str | os.PathLike = DEFAULT_REPO, device: str | torch.device | None = None,
                        *, revision: str | None = None, cache_dir: str | os.PathLike | None = None,
                        token: str | bool | None = None, **options) -> "TTS":
        """Load the model from a local folder or a Hugging Face repository.

        device: ``"cpu"``, ``"cuda"``, ... ; None picks CUDA if available, else the CPU.
        revision, cache_dir, token: passed to ``huggingface_hub.snapshot_download`` (only for repositories).
        Further keyword options go to :class:`TTS` (``uniform_rate``, ``low_shelf_db``, ``peak_limit``,
        ``watermark``)."""
        folder = Path(repo_or_dir).expanduser()
        if not folder.is_dir():
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                raise ImportError(f"{repo_or_dir!r} is not a local folder; to download it, install huggingface_hub "
                                  "(pip install huggingface_hub)") from None
            folder = Path(snapshot_download(repo_id=str(repo_or_dir), revision=revision, cache_dir=cache_dir,
                                            token=token, allow_patterns=list(WEIGHT_FILES)))
        missing = [f for f in WEIGHT_FILES if not (folder / f).is_file()]
        if missing:
            raise FileNotFoundError(f"{folder} lacks {', '.join(missing)}")
        config = ArticulinoConfig.from_file(folder / "config.json")
        model = AcousticModel(config)
        model.load_state_dict(load_file(folder / "model.safetensors")[0], strict=True)
        codec = Codec.from_config(config.codec, config.latents.dim)
        codec.load_state_dict(load_file(folder / "codec.safetensors")[0], strict=True)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(model, codec, config, device, **options)

    def _autocast(self):
        """bfloat16 autocast on CUDA for the acoustic model (as in the service the model was evaluated with; the
        codec always runs in float32); float32 everywhere else."""
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    # ------------------------------------------------------------------------------------------- voice
    @torch.inference_mode()
    def set_voice(self, voice: str | os.PathLike | np.ndarray | torch.Tensor | Voice,
                  sample_rate: int | None = None) -> Voice:
        """Use this voice from now on.

        voice: path of an audio file, a mono array of samples (``sample_rate``, default 24000), or a :class:`Voice`
        returned earlier. 3 to 8 seconds of clean speech work best; longer recordings are cut to 8 seconds, other
        sample rates are resampled to 24 kHz. Only clone voices of people who agreed to it."""
        if isinstance(voice, Voice):
            self.voice = voice
            return voice
        if isinstance(voice, (str, os.PathLike)):
            wav, sr = load_audio(voice)
        else:
            wav = voice.detach().cpu().numpy() if isinstance(voice, torch.Tensor) else np.asarray(voice)
            wav = wav.astype(np.float32)
            if wav.ndim == 2:
                wav = wav.mean(0 if wav.shape[0] < wav.shape[1] else 1)
            sr = self.sample_rate if sample_rate is None else int(sample_rate)
        if wav.ndim != 1:
            raise ValueError("the voice must be mono audio")
        if sr != self.sample_rate:
            wav = resample(wav, sr, self.sample_rate)
        hop = self.config.latents.hop_length
        n = min(len(wav), int(self.config.voice_max_seconds * self.sample_rate)) // hop * hop
        seconds = n / self.sample_rate
        if seconds < MIN_VOICE_SECONDS:
            raise ValueError(f"the voice recording is too short ({len(wav) / self.sample_rate:.2f} s); use 3 to 8 s")
        if seconds < 3.0:
            warnings.warn(f"voice recording of {seconds:.1f} s; 3 to 8 s of clean speech work best", stacklevel=2)
        if not np.isfinite(wav[:n]).all():
            raise ValueError("the voice recording contains NaN or infinite samples")
        x = torch.from_numpy(np.ascontiguousarray(wav[:n]))[None].to(self.device)
        latents = (self.codec.encoder(x).transpose(1, 2) - self._mean) / self._std      # float32 on every device
        with self._autocast():
            state = self.model.prepare_voice(latents)
        self.voice = Voice(state=state, rate=self._natural_rate(state) if self.uniform_rate else None, seconds=seconds)
        return self.voice

    def _natural_rate(self, state: VoiceState) -> float:
        """Median letters per second the duration predictor gives this voice on neutral sentences."""
        rates: list[float] = []
        lang = torch.zeros(1, dtype=torch.long, device=self.device)
        for sentence in RATE_CALIBRATION:
            ids, _ = encode(sentence)
            t = torch.tensor([ids], device=self.device)
            mask = torch.ones_like(t, dtype=torch.bool)
            with self._autocast():
                h = self.model.text_encoder(t, lang, mask)
                d = self.model.duration_predictor(h, state.vector, mask).float().exp()
            rates += letter_rate(d, t)[0]
        return float(np.median(rates))

    # ------------------------------------------------------------------------------------------- speaking
    def say(self, text: str, path: str | os.PathLike | None = None, speed: float = 1.0, seed: int = 0,
            language: str = "de", normalize: bool = True) -> Speech:
        """Speak a whole text.

        The text goes through the German text frontend (numbers, dates, units, abbreviations, …; only for
        ``language="de"`` and with ``normalize=True``), is split into sentences, and every sentence is generated on
        its own with the same ``seed``; sentences are joined with 0.15 s of silence. ``speed`` (0.7 to 1.6) scales
        the speaking rate. With ``path``, the audio is also written as a WAV file."""
        speed, seed = self._check(speed, seed, language)
        if normalize and language == "de":
            text = _normalize_german(text).strip()
        pieces = carry_language_tags(split_sentences(text, True, self.config.sentences.max_chars,
                                                     self.config.sentences.min_letters)[0])
        gap = np.zeros(int(round(SENTENCE_PAUSE * self.sample_rate)), dtype=np.float32)
        chunks: list[np.ndarray] = []
        for piece in pieces:
            audio = self._speak(piece, speed, seed, language)
            if audio.size:
                if chunks:
                    chunks.append(gap)
                chunks.append(audio)
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        speech = Speech(audio=audio, sample_rate=self.sample_rate, duration=len(audio) / self.sample_rate, seed=seed)
        if path is not None:
            speech.save(path)
        return speech

    def stream(self, text: str | Iterable[str], speed: float = 1.0, seed: int = 0, language: str = "de",
               normalize: bool = True) -> Iterator[np.ndarray]:
        """Speak text while it is still arriving; yields one float32 chunk per sentence.

        ``text`` is a string or any iterable of text pieces (e.g. tokens from a language model). Incoming text is
        collected until a sentence is complete, then that sentence is spoken at once. Sentences with fewer than 20
        letters wait for the next one; very long ones are split at commas."""
        speed, seed = self._check(speed, seed, language)          # fail here, not at the first next()
        return self._stream([text] if isinstance(text, str) else text, speed, seed, language, normalize)

    def _stream(self, pieces: Iterable[str], speed: float, seed: int, language: str,
                normalize: bool) -> Iterator[np.ndarray]:
        normalizer = _TaggedStreamNormalizer() if normalize and language == "de" else None
        buffer = SentenceBuffer(self.config.sentences.max_chars, self.config.sentences.min_letters)
        for piece in pieces:
            if normalizer is not None:
                piece = normalizer.push(piece)
            if piece:
                for sentence in buffer.push(piece):
                    audio = self._speak(sentence, speed, seed, language)
                    if audio.size:
                        yield audio
        tail = normalizer.flush() if normalizer is not None else ""
        for sentence in buffer.flush(tail):
            audio = self._speak(sentence, speed, seed, language)
            if audio.size:
                yield audio

    def _check(self, speed: float, seed: int, language: str) -> tuple[float, int]:
        if self.voice is None:
            raise RuntimeError("no voice set: call tts.set_voice('voice.wav') first")
        if language not in self.config.text.languages:
            raise ValueError(f"language must be one of {sorted(self.config.text.languages)}, not {language!r}")
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be a non-negative integer")
        speed = float(speed)
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be a positive number")
        return min(MAX_SPEED, max(MIN_SPEED, speed)), seed

    def _speak(self, sentence: str, speed: float, seed: int, language: str) -> np.ndarray:
        ids, langs = encode(sentence, language)
        if not ids:
            return np.zeros(0, dtype=np.float32)
        audio = self._postprocess(*self.synthesize(ids, langs, speed, seed))
        if self.watermark is not None:
            audio = np.asarray(self.watermark(audio, self.sample_rate), dtype=np.float32)
        return audio

    @torch.inference_mode()
    def synthesize(self, ids: list[int], languages: list[int] | int = 0, speed: float = 1.0, seed: int = 0,
                   noise: list[torch.Tensor] | None = None) -> tuple[np.ndarray, int]:
        """Generate one sentence from character ids (no frontend, no splitting, no post-processing).

        languages: one id for the whole sentence or one per character. noise: one tensor (1, new frames, 96) per
        sampling step instead of drawing it from ``seed`` (for tests). Returns (raw decoder audio including the
        context frames in front, number of context samples to drop)."""
        if self.voice is None:
            raise RuntimeError("no voice set: call tts.set_voice('voice.wav') first")
        dev, cfg, voice = self.device, self.config, self.voice
        state = voice.state
        p = state.prompt_frames
        units = cfg.sampler.duration_units_per_frame
        text = torch.tensor([[cfg.text.pad_id] + list(ids)], device=dev)
        mask = torch.tensor([[False] + [True] * len(ids)], device=dev)
        if isinstance(languages, int) or len(set(languages)) == 1:
            lang = torch.tensor([languages if isinstance(languages, int) else languages[0]], device=dev)
        else:
            lang = torch.tensor([[languages[0]] + list(languages)], device=dev)
        with self._autocast():
            h = self.model.text_encoder(text, lang, mask)
            log_d = self.model.duration_predictor(h, state.vector, mask)
            d = (log_d.float().exp() * mask).clamp(max=cfg.duration_predictor.max_units) / speed
            if self.uniform_rate:
                if voice.rate is None:                  # voice prepared by an instance without uniform_rate
                    voice.rate = self._natural_rate(state)
                d = letter_rate(d, text, start=1, target=voice.rate * speed)[1]
            d[:, 0] = units * p
            frames = int(torch.ceil(d.sum(1) / units).long().clamp(min=1).item())
            if frames <= p:
                return np.zeros(0, dtype=np.float32), 0
            frame_cond = expand_to_frames(h, d, frames, units)[:, p:]
            if noise is None:
                noise = draw_noise(seed, frames, p, len(cfg.sampler.timesteps), cfg.latents.dim)
            x1 = sample(self.model.dit, frame_cond, state, cfg.sampler.timesteps, noise).float()
        ctx = min(p, cfg.sampler.decoder_context_frames)
        z = torch.cat([state.prompt[:, p - ctx:].float(), x1], 1)
        wav = self.codec.decoder((z * self._std + self._mean).transpose(1, 2))         # float32 on every device
        return wav[0].float().cpu().numpy(), ctx * cfg.latents.hop_length

    def _postprocess(self, wav: np.ndarray, skip: int) -> np.ndarray:
        """Low-shelf over the whole decoder output (so the filter has settled), drop the context, limit the peak."""
        if self._shelf is not None and len(wav) > skip:
            wav = self._shelf(wav)
        wav = np.asarray(wav[skip:], dtype=np.float32)
        return limit_peak(wav, self.peak_limit).astype(np.float32, copy=False)
