# Articulino inference specification

This document specifies the computation that every Articulino engine implements: the PyTorch reference package (this
repository), [articulino-onnx](https://github.com/eigenhand/articulino-onnx),
[articulino.cpp](https://github.com/eigenhand/articulino.cpp) and
[articulino-mlx](https://github.com/eigenhand/articulino-mlx). Where this text and the code of the PyTorch package
disagree, the code is authoritative, and the disagreement is a bug in this document.

All sizes below are those of the released model; every one of them is also in `config.json`.

## 1. Weights

| File | Content |
|---|---|
| `model.safetensors` | acoustic model: text encoder, voice encoder, duration predictor, DiT (66.3 M parameters) |
| `codec.safetensors` | latent encoder (21.5 M) and latent decoder (21.7 M) |
| `config.json` | every size, the latent mean and standard deviation (96 values each), the four sampling timesteps (0, 0.25, 0.5, 0.75), sample rate, character set, post-processing |

Tensor names, shapes and their origin in the training checkpoints: [weights.md](weights.md).

## 2. Text

- Character set `" abcdefghijklmnopqrstuvwxyzäöüß.,?!-'\":;()0123456789"`; id = index + 1; 0 is padding and the
  placeholder for the voice prompt. Vocabulary size 53.
- Character normalization: lower case, a fixed replacement table (quotes, dashes, ellipsis, …), accents removed with
  NFKD except for ä ö ü ß, unknown characters dropped, runs of whitespace collapsed to one space
  (`articulino.text.normalize_characters`).
- Language: 0 = German, 1 = English, either per sentence (B,) or per letter (B, N). Inline markup
  `[en]…[/en]` / `[de]…[/de]` switches the language of the letters inside.
- Before the model, German text goes through the German text frontend (`articulino.frontend`): numbers, dates, times,
  money, units, abbreviations, formulas, phone numbers, postcodes, URLs and spelled-out letters become words. The
  model has practically never seen digits in training.
- Sentences are split at sentence ends, and long sentences at commas and other breaks, so that no piece exceeds 180
  characters (`articulino.sentences`).

## 3. Per sentence (inference)

Inputs: a voice recording (24 kHz mono) and a sentence.

1. **Voice (once per voice).** Cut the waveform to a multiple of 960 samples and at most 8 s → log-mel → latent
   encoder → `mu` (96, R) → `ref = (mu.T − mean) / std` (R, 96). The prompt is `prompt = ref[:125]` (5 s).
2. **Voice features.** `tokens, g = voice_encoder(ref, mask)` (32 × 768 and 768). Voice cache: for every DiT block,
   the self-attention keys and values of the prompt frames, shape (heads 6, 125, 64).
3. **Text.** ids = `[0] + ids(sentence)`, mask `[False] + [True] * n`: position 0 is the placeholder for the prompt and
   carries no text. `h = text_encoder(ids, language, mask)`; the placeholder is masked out.
4. **Durations.** `d = exp(duration_predictor(h, g, mask)) * mask`, clamped to at most 400, divided by `speed`;
   `d[0] = 4 * 125` (the prompt). Unit: 10 ms; 4 units = 1 latent frame (25 Hz). Optionally every sentence is brought
   to the voice's natural letter rate (`uniform_rate`, on by default): the rate is measured once per voice on six
   calibration sentences, and the durations of each sentence are scaled to it.
5. **Frame conditioning.** `L = ceil(sum(d) / 4)`; `hf = expand(h, d, L)`: the weight of letter ℓ on frame j is the
   overlap of [start_ℓ, end_ℓ) with [4j, 4j + 4), divided by 4.
6. **Sampling: four steps, only the new frames (L − 125).**
   `x = noise_0`; for k = 0…3 with t_k ∈ {0, 0.25, 0.5, 0.75}:
   `v = dit(x, t_k, hf[125:], tokens, g, cache)`; `x1 = x + (1 − t_k) · v`;
   if k < 3: `x = (1 − t_{k+1}) · noise_{k+1} + t_{k+1} · x1`. The result is `x1` of step 3. No classifier-free guidance.
7. **Decoder.** `z = concat(prompt[−12:], x1)` (12 frames of context), de-normalize `z · std + mean`, transpose to
   (96, T) → latent decoder → waveform; drop the first 12 · 960 samples.
8. **Post-processing.** Low-shelf filter +2 dB at 180 Hz (RBJ cookbook, shelf slope S = 0.7), then scale the sentence
   down if its peak exceeds 0.95. `say()` joins sentences with 150 ms of silence; `stream()` yields one block per
   sentence.

Noise: engines may use their own random numbers, but for the equality tests the noise must be injectable
(`tests/vectors/*.npz`: `noise_0` … `noise_3`). The PyTorch package draws it with `torch.Generator(seed)` on the CPU
over all frames (prompt included); the MLX port reproduces that generator bit for bit.

## 4. Building blocks (exact computation)

- **Linear, LayerNorm:** as in PyTorch. LayerNorm eps 1e-5, except in the ConvNeXt blocks of the codec (1e-6).
  AdaNorm: LayerNorm without weights (eps 1e-5), then `x · (1 + γ) + β` with `[γ, β] = Linear(c)`.
- **GELU:** exact (erf), not the tanh approximation. **SiLU** in the time MLP.
- **Attention:** `q = Linear(x)`, `[k, v] = Linear(ctx)` (one 2·dim matrix, k first), separate heads,
  `softmax(q·kᵀ / √dh) · v`, output Linear. Masks: boolean masks become −1e9; in the DiT path with the prompt the mask
  is additive (0 / −30000).
- **RoPE:** frequencies `1 / 10000^(2i/dh)`, positions 0…L−1, **interleaved pairs** (even/odd indices:
  `x1 = x[0::2], x2 = x[1::2]`, result `x1·cos − x2·sin, x1·sin + x2·cos` interleaved again). Self-attention only.
  In the cached path the new frames start at position 125.
- **Time embedding:** `f = exp(−ln(10000) · arange(128) / 128)`, `w = t · 1000 · f`, `concat(sin w, cos w)` (256) →
  Linear → SiLU → Linear.
- **Text encoder:** embedding (53 × 512, padding 0) + language embedding (2 × 512); 6 pre-norm blocks (attention with
  RoPE, 8 heads; feed-forward ×4 with GELU); LayerNorm; output × mask.
- **Voice encoder:** Linear 96 → 512, 4 blocks, 32 learned queries, 2 × (LN, cross-attention to the blocks, LN, FF),
  LN, `tokens = Linear(512 → 768)`, `vector = MLP(mean(queries))`.
- **Duration predictor:** `x = (h + Linear(g)) · mask`, 3 × (Conv1d k3 p1 → ReLU → LN) · mask, Linear → log duration.
- **DiT:** d = 384, 8 blocks, 6 heads, dh = 64. `tokens` and `g` are first projected 768 → 384. Input:
  `Linear(concat(x_t, prompt, prompt_mask))` (2·96 + 1 → 384) + `Linear(hf)` (512 → 384). Block: AdaNorm →
  self-attention (RoPE) → residual; LN → cross-attention to the voice tokens → residual; AdaNorm → FF → residual.
  Final: AdaNorm, Linear 384 → 96. Conditioning c per frame: `time(t) + g`. In the cached path the new frames attend
  to the cached prompt keys/values and to themselves, without a mask.
- **Log-mel (encoder input):** like torchaudio `MelSpectrogram(sample_rate 24000, n_fft 1024, win_length 960,
  hop 240, n_mels 100, f_min 0, f_max 12000, power 1.0, center True)`: **periodic** Hann window centred in 1024,
  reflect padding, HTK mel scale, no filter normalization; then `log(clamp(·, 1e-5))`, cut to exactly T / 240 frames
  (torchaudio returns one more).
- **Latent encoder:** Conv1d 100 → 512 k7 p3; [4 × ConvNeXt (inner 1536), Conv1d k4 s2 p1] × 2; 4 × ConvNeXt;
  LN (1e-6); Linear 512 → 96 (`mu`).
- **ConvNeXt:** depthwise Conv1d k7 p3 → LN (eps 1e-6) → Linear ×3 → GELU → Linear → `x + γ · y` (γ per channel).
- **Latent decoder:** Conv1d 96 → 512 k3 p1 → ConvTranspose1d k8 s4 p2 (25 → 100 Hz) → 12 × ConvNeXt → LN (1e-6) →
  Linear 512 → 962 → `log_magnitude, phase` (481 each) → `magnitude = min(exp(log_magnitude), 100)` → complex
  spectrum `magnitude · e^{i·phase}` → **inverse STFT** (n_fft 960, hop 240, periodic Hann 960, center True,
  length = frames · 960; overlap-add divided by the sum of squared windows, as `torch.istft`).

## 5. Equality tests

`tests/vectors/de_sentence.npz` and `en_sentence.npz` (with a `.json` describing sentence, language, timesteps,
frame counts and shapes, and a `.wav` to listen to) were computed with the original inference code on the CPU in
float32. They hold every intermediate step: `voice_wav`, `voice_logmel`, `voice_mu`, `voice_ref`, `text_ids`,
`text_mask`, `text_h`, `voice_tokens`, `voice_vector`, `duration_log`, `durations`, `frame_cond`,
`cache_k{i}` / `cache_v{i}`, `noise_0..3`, `step{k}_in` / `_velocity` / `_x1`, `latents_norm`, `decoder_in`,
`decoder_out`, `audio`.

Tolerances for a float32 engine: every building block, fed with the reference inputs, within
`max|Δ| / max|ref| < 1e-3`; end-to-end audio SNR above 40 dB. Reduced-precision versions (float16, 8-bit, 4-bit)
report the SNR per block and are checked by ear and with a speech recognizer.

Speed is measured the same way in every engine: ten reference sentences, one voice with its cache built beforehand,
one warm-up sentence, then the whole chain per sentence; real time factor = audio seconds per compute second.
