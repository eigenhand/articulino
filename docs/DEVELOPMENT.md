# Development notes: PyTorch package

State of 15 October 2026. Package in `src/articulino` (src layout), tests in `tests/`, tools in `tools/`.

## What the package does

- Public API: `TTS.from_pretrained(repo_or_dir, device=None)`, `tts.set_voice(path | array | Voice)`,
  `tts.say(text, path=None, speed=1.0, seed=0, language="de", normalize=True) -> Speech(audio, sample_rate, duration, seed)`,
  `tts.stream(text | iterable)` (one float32 block per sentence), CLI `articulino say "…" --voice v.wav -o out.wav`
  and `articulino normalize "…"`, and a `watermark=None` hook (call site only; no watermark is built in).
- Modules: `text` (character set, `[en]…[/en]`), `model` (text encoder, voice encoder, duration predictor, DiT with
  voice cache), `codec` (log-mel without torchaudio, latent encoder, decoder with inverse STFT), `sampler` (4 steps,
  injectable noise, letter rate), `sentences` (sentence splitting), `postprocess` (low-shelf +2 dB at 180 Hz without
  SciPy, peak limit 0.95), `audio` (reading, resampling like torchaudio), `frontend` (German text frontend).
- Dependencies: torch, numpy, soundfile, num2words (optionally huggingface_hub). Own safetensors reader, own mel
  filter bank (bit-identical to torchaudio), own IIR filter (≈1e-13 from `scipy.signal.sosfilt`).
- Behaviour matches the service the model was evaluated with: the same voice cache, one uniform letter rate per voice
  (calibration sentences), noise from `torch.Generator(seed)` over all frames including the prompt, decoder with 12
  context frames, shelf over the context, peak 0.95, speed clamped to 0.7–1.6, `say()` = frontend → sentence split →
  150 ms pauses, `stream()` = streaming frontend → sentence split, the same seed for every sentence.

## Export

`tools/export_safetensors.py` turns the two training checkpoints into `model.safetensors` (391 tensors, 66.27 M),
`codec.safetensors` (234 tensors, 43.18 M), `config.json` and `tensor_mapping.json` (old → new name per tensor; table
in [weights.md](weights.md)). Left out, because only training uses them: the "no text" / "no voice" embeddings for
classifier-free guidance, the variance half of the encoder output, and the mel buffers (recomputed exactly in code).

## Equality with the reference vectors (CPU, float32, noise from the vectors)

max|Δ| / max|ref| per block (limit 1e-3), every block fed with the reference inputs:

| Block | de_sentence | en_sentence |
|---|---|---|
| log-mel | 7.2e-8 | 7.2e-8 |
| encoder mu | 2.5e-7 | 2.5e-7 |
| normalized latents | 0 | 0 |
| text encoder | 0 | 0 |
| voice tokens | 9.3e-7 | 9.3e-7 |
| voice vector | 2.4e-7 | 2.4e-7 |
| log duration / durations | 0 / 0 | 0 / 0 |
| frame conditioning | 6.6e-8 | 6.7e-8 |
| voice cache (k, v, all 8 blocks) | 9.9e-7 | 9.9e-7 |
| DiT velocity, steps 0–3 (max) | 2.0e-6 | 7.3e-7 |
| 4-step latents | 2.2e-6 | 1.1e-6 |
| decoder (inverse STFT) | 2.1e-6 | 2.9e-6 |
| **end-to-end audio (SNR)** | **90.7 dB** | **75.1 dB** |

End-to-end = voice WAV → encoder → voice cache → text → durations → sampler → decoder through the public chain, with
only the noise taken from the vectors.

During development, `say()` was also compared with a re-run of the original service path (voice cache, letter rate,
seeded noise, shelf, peak limit, sentence splitting, 150 ms pauses; four cases with speed 0.8/1.0/1.3 and seeds
0/3/7/42): same length, SNR 92.5–108.9 dB, identical base letter rate of the voice (18.587 letters/s).

## German text frontend (`articulino.frontend`)

- A port of the original frontend with English names; rules, regular expressions and tables unchanged (tables as
  `abbreviations.json` and `units.json`, same entries in the same order).
- All 129 sentences of the pronunciation suite give exactly the original output (`tests/data/pronunciation_cases.tsv`).
- During development the port was compared with the original module on 6,622 texts (every unit and abbreviation in
  context, random mixes of numbers, dates, formulas, Markdown and addresses) and on 3,000 random stream chunkings:
  no difference.
- New in the package: `[en]…[/en]` spans bypass the German frontend (otherwise `[en]a[/en]` would be read "ah" and
  `[en]PDF[/en]` "peh deh eff").
- `language="en"`: the German frontend does not run, only character normalization; write numbers in English text as
  words.

## Tests (`pytest`)

Building blocks against the vectors, end to end, the frontend against the pronunciation suite, characters, sentence
splitting (3,000 random buffers), the post-processing filter against SciPy, resampling against torchaudio, the weight
files (own reader = safetensors library), and the public API (`say`, `stream`, seeds, speed, languages, options,
48 kHz stereo voices, CLI). Tests that need weights, vectors, SciPy or torchaudio are skipped without them.

## Real-time factor (CPU, 10 sentences, voice cache built beforehand)

Measured with `tools/rtf.py` on the efficiency cores of an Intel Core i9-14900K (in a VM), best of 3 runs; time to the
first sentence in brackets. "Reference" is the original evaluation script, "raw" the same work in this package,
`say()` adds the frontend and the uniform rate.

| Threads | reference | `raw` | `say()` |
|---|---|---|---|
| 12 | 26.5× (0.17 s) | **31.6×** (0.14 s) | **31.5×** (0.14 s) |
| 8 | 22.6× (0.20 s) | **27.2×** (0.17 s) | **27.6×** (0.16 s) |
| 4 | 14.6× (0.31 s) | **17.6×** (0.26 s) | **17.6×** (0.25 s) |
| 2 | 8.4× (0.53 s) | **10.3×** (0.44 s) | **10.3×** (0.43 s) |

The package is 15–23 % faster than the reference because it computes once per voice what the reference recomputes per
sentence (voice encoder, cross-attention keys/values of the voice tokens), and the AdaNorm modulation once per step
instead of per frame. With 12 threads, a single busy core slows a run down to 2–3× (OpenMP stragglers).

## Listening check

`tools/listen_check.py`: the four sample sentences of the model card with `say()`, seed 0, transcribed by
Parakeet-TDT 0.6B v3. Mean CER 0.011. The one error is an extra "Uhr" after "Königstraße zwölf" at seed 0 (correct
with seed 1); it is model behaviour, not a package error, since `say()` matches the original service path above
90 dB, and it comes from the uniform letter rate (correct without it).

## Clean install

A wheel (`articulino-1.0.0-py3-none-any.whl`, includes the JSON tables) installed into a fresh environment with only
CPU PyTorch pulls in numpy, soundfile and num2words, and no SciPy, torchaudio, safetensors or huggingface_hub.

## Open points

- The CUDA path (bfloat16 autocast for the acoustic model, codec in float32) and Apple MPS are untested;
  `device=None` uses the CPU on Apple silicon (use [articulino-mlx](https://github.com/eigenhand/articulino-mlx)).
- The original service repeated a sentence with seed + 1 when it detected a whistle or latent outliers; the package
  does not have this guard (the C++ and ONNX engines do).
- Numbers and abbreviations inside `[en]…[/en]` are not expanded (on purpose, see above).
