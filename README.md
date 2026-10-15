# articulino

Reference Python package for **Articulino-DE-EN-Realtime-Stream-TTS**: small, fast German and English text to speech
with zero-shot voice cloning. A 4-step distilled flow-matching model on 25 Hz audio latents, plain PyTorch, runs in
real time on a single CPU core.

Model card and weights: <https://huggingface.co/eigenhand/Articulino-DE-EN-Realtime-Stream-TTS>
(Apache 2.0, code and weights).

## Install

```bash
pip install "articulino[hub] @ git+https://github.com/eigenhand/articulino"
```

Or from a clone: `pip install ".[hub]"`.

Requirements: Python 3.10+, PyTorch 2.1+. `huggingface_hub` is only needed to download the weights; a local folder
works without it. No torchaudio, SciPy or `safetensors` package is needed.

## Quickstart

```python
from articulino import TTS

tts = TTS.from_pretrained("eigenhand/Articulino-DE-EN-Realtime-Stream-TTS")   # or a local folder; GPU if there is one
tts.set_voice("voice.wav")                          # 3–8 s of clean speech from someone who agreed to this
speech = tts.say("Guten Morgen! Der Termin ist um 15:30 Uhr.", path="morgen.wav")
print(speech.duration, speech.sample_rate)          # seconds, 24000
```

`say()` returns a `Speech` with `audio` (float32 NumPy array in [-1, 1]), `sample_rate`, `duration` (seconds) and
`seed`. With `path`, it also writes a 16-bit WAV file.

| Option | Values | Default |
|---|---|---|
| `language` | `"de"`, `"en"` | `"de"` |
| `speed` | 0.7 to 1.6 (values outside are clamped) | 1.0 |
| `seed` | non-negative integer; the same seed gives the same audio | 0 |
| `normalize` | run the German text frontend (German only) | `True` |
| `path` | an audio file to write (`.wav`, `.flac`, …) | none |

Words in the other language can be marked inside a sentence: `"Das [en]Meeting[/en] ist verschoben."`. Every
letter then carries its language. Tagged spans skip the German frontend (so `[en]a[/en]` stays "a" and is not
spelled "ah"); write numbers inside them in words.

### `stream()`: speak while the text is still coming

```python
import sounddevice as sd                          # pip install sounddevice

def llm_tokens():                                 # any iterator of text pieces, e.g. tokens from a language model
    yield from ["Guten Mor", "gen! Heute ", "regnet es ab 18 Uhr. ", "Nimm einen Schirm mit."]

with sd.OutputStream(samplerate=24000, channels=1, dtype="float32") as speaker:
    for chunk in tts.stream(llm_tokens()):
        speaker.write(chunk)                      # one chunk per sentence
```

`stream()` takes a string or any iterable of text pieces and yields one float32 chunk per sentence, as soon as the
sentence is complete. Sentences with fewer than 20 letters wait for the next one; sentences longer than 180 characters
are split at a comma. The chunks are exactly the sentences of `say()` (which joins them with 0.15 s of silence).

### Command line

```bash
articulino say "Guten Morgen! Der Termin ist um 15:30 Uhr." --voice voice.wav -o morgen.wav
articulino say - --voice voice.wav -o out.wav --speed 1.2 --seed 3 < text.txt
articulino normalize "Die Rechnung über 87,40 € ist am 15.11. fällig."
```

`--model` takes a local folder or a repository (default: the release above), `--device` `cpu`/`cuda`,
`--threads` the number of CPU threads, `--language de|en`, `--no-normalize` skips the German frontend.

## What happens to a text

1. **German frontend** (`articulino.frontend.normalize`, only for `language="de"` with `normalize=True`): numbers,
   dates, times, money, units, phone numbers, formulas (LaTeX and Unicode), e-mail and web addresses, Markdown and
   emoji are written out or removed, abbreviations are expanded (z. B. → zum Beispiel) and upper-case abbreviations
   are spelled with German letter names (PDF → „peh deh eff“). The model reads letters and never saw digits.
2. **Sentences**: split at `. ! ? … : ;`; short ones are joined, long ones split at commas.
3. **Per sentence**: characters → text encoder → durations (10 ms units) → per-frame text features → DiT, 4 steps
   from noise, continuing from the first 5 s of the voice recording (cached once per voice) → decoder (iSTFT) →
   24 kHz audio.
4. **Post-processing**: +2 dB low-shelf at 180 Hz, then the chunk is scaled down if its peak is above 0.95.

By default, every sentence is brought to the voice's natural speaking rate × `speed` (measured once per voice on six
neutral sentences), so the tempo stays steady from sentence to sentence. Options of `from_pretrained` / `TTS`:

| Option | Meaning | Default |
|---|---|---|
| `uniform_rate` | steady speaking rate per voice (otherwise the duration predictor's rate as is) | `True` |
| `low_shelf_db` | gain of the low-shelf at 180 Hz; 0 switches it off | 2.0 |
| `peak_limit` | peak above which a sentence is scaled down; `None` switches it off | 0.95 |
| `watermark` | reserved hook `f(audio, sample_rate) -> audio`, applied to every chunk; nothing is built in | `None` |

`set_voice()` accepts a file path, a mono NumPy array or tensor (`sample_rate=`, default 24000) or a `Voice` returned
earlier, so several voices can be prepared once and switched without recomputing. Other sample rates are resampled to
24 kHz; recordings are cut to 8 s; below 3 s a warning is given.

On CUDA, the acoustic model runs in bfloat16 autocast and the codec in float32 (as in the service the model was
evaluated with); on the CPU everything runs in float32.
Noise is always drawn on the CPU, so a seed gives the same audio on every device type up to arithmetic precision.

## Files of a release

| File | Content |
|---|---|
| `config.json` | every size, latent mean/std, sampling times, sample rate, character set, post-processing |
| `model.safetensors` | acoustic model: text encoder, voice encoder, duration predictor, DiT (66.3 M) |
| `codec.safetensors` | latent encoder (voice recordings) and decoder (21.5 M + 21.7 M) |

Tensor names and the mapping from the training checkpoints: [docs/weights.md](docs/weights.md).
`tools/export_safetensors.py` builds these files from the training checkpoints.

## Tests

```bash
pip install -e ".[hub,test]"
hf download eigenhand/Articulino-DE-EN-Realtime-Stream-TTS --local-dir build/release   # or ARTICULINO_MODEL_DIR=…
python -m pytest
```

The tests check every building block against the reference vectors in `tests/vectors/` (computed with the original
inference code; relative error below 1e-3, end-to-end audio above 40 dB SNR), the German frontend against the
expected readings of the 129-sentence pronunciation suite, sentence splitting, the post-processing filter and the
public API. `tools/rtf.py` measures the real-time factor, `tools/listen_check.py` runs a listening check through a
speech recognizer.

The computation is specified step by step in [docs/SPEC.md](docs/SPEC.md); the other engines implement the same
specification and are tested against the same vectors. Measurements and engineering notes:
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## Other engines

| Engine | Code | Weights |
|---|---|---|
| ONNX Runtime (CPU, GPU, browser) | [eigenhand/articulino-onnx](https://github.com/eigenhand/articulino-onnx) | [Articulino-DE-EN-Realtime-Stream-TTS-ONNX](https://huggingface.co/eigenhand/Articulino-DE-EN-Realtime-Stream-TTS-ONNX) |
| C++ / GGUF (ggml, CPU and Vulkan) | [eigenhand/articulino.cpp](https://github.com/eigenhand/articulino.cpp) | [Articulino-DE-EN-Realtime-Stream-TTS-GGUF](https://huggingface.co/eigenhand/Articulino-DE-EN-Realtime-Stream-TTS-GGUF) |
| MLX (Apple silicon) | [eigenhand/articulino-mlx](https://github.com/eigenhand/articulino-mlx) | [Articulino-DE-EN-Realtime-Stream-TTS-MLX](https://huggingface.co/eigenhand/Articulino-DE-EN-Realtime-Stream-TTS-MLX) |

## Responsible use

Only clone voices with consent, tell people they hear synthetic speech, and mark generated audio where the law asks
for it (EU AI Act, Art. 50). The weights contain no watermark. See the model card for details.

## License

Apache License 2.0.
