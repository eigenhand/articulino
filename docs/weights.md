# Weights and configuration

A release folder holds three files:

| File | Tensors | Parameters | Content |
|---|---|---|---|
| `model.safetensors` | 391 | 66.27 M | acoustic model: text encoder, voice encoder, duration predictor, DiT (student) |
| `codec.safetensors` | 234 | 43.18 M | latent encoder (log-mel → latents) and decoder (latents → iSTFT → audio) |
| `config.json` | – | – | sizes, latent statistics, sampling times, sample rate, character set, post-processing |

All tensors are float32, little endian. They are written by `tools/export_safetensors.py` from the training
checkpoints `schueler-dmd1-020000.pt` (student after 20,000 DMD2 steps; key `ema`) and `vocoder-d1-020000.pt`
(autoencoder with the fine-tuned decoder d1; key `ae`). The export also writes `tensor_mapping.json` with the full
old → new name of every tensor.

## Not exported (training only)

| Old name | Why |
|---|---|
| `null_text` | "no text" embedding for classifier-free guidance; the student runs without guidance |
| `ref.null_tokens`, `ref.null_vektor` | "no voice" tokens/vector for guidance |
| rows 96–191 of `encoder.aus.weight` / `.bias` | log-variance head of the VAE encoder; only the mean (rows 0–95) is used |
| `encoder.mel.mel.spectrogram.window`, `encoder.mel.mel.mel_scale.fb` | Hann window and mel filter bank; recomputed exactly in code (`articulino.codec.mel_filterbank`) |

## Name mapping

`{i}` is a layer index (kept as is unless stated). Every `…` stands for both `weight` and `bias`.

### `model.safetensors`

| Training name | Release name | Shape |
|---|---|---|
| `text.emb.weight` | `text_encoder.char_embedding.weight` | 53 × 512 |
| `text.sprache.weight` | `text_encoder.language_embedding.weight` | 2 × 512 |
| `text.bloecke.{i}.n1.…` | `text_encoder.layers.{i}.attn_norm.…` | 512 |
| `text.bloecke.{i}.att.q.…` | `text_encoder.layers.{i}.attn.q_proj.…` | 512 × 512 |
| `text.bloecke.{i}.att.kv.…` | `text_encoder.layers.{i}.attn.kv_proj.…` | 1024 × 512 (keys rows 0–511, values 512–1023) |
| `text.bloecke.{i}.att.aus.…` | `text_encoder.layers.{i}.attn.out_proj.…` | 512 × 512 |
| `text.bloecke.{i}.n2.…` | `text_encoder.layers.{i}.ff_norm.…` | 512 |
| `text.bloecke.{i}.ff.a.…` / `ff.b.…` | `text_encoder.layers.{i}.ff.up.…` / `ff.down.…` | 2048 × 512 / 512 × 2048 |
| `text.norm.…` | `text_encoder.final_norm.…` | 512 |
| `ref.ein.…` | `voice_encoder.input_proj.…` | 512 × 96 |
| `ref.bloecke.{i}.*` | `voice_encoder.layers.{i}.*` (same scheme as the text encoder) | |
| `ref.fragen` | `voice_encoder.queries` | 32 × 512 |
| `ref.pool.{j}.0.…` | `voice_encoder.pool.{j}.attn_norm.…` | 512 |
| `ref.pool.{j}.1.{q,kv,aus}.…` | `voice_encoder.pool.{j}.attn.{q_proj,kv_proj,out_proj}.…` | |
| `ref.pool.{j}.2.…` | `voice_encoder.pool.{j}.ff_norm.…` | 512 |
| `ref.pool.{j}.3.{a,b}.…` | `voice_encoder.pool.{j}.ff.{up,down}.…` | |
| `ref.norm.…` | `voice_encoder.final_norm.…` | 512 |
| `ref.aus.…` | `voice_encoder.token_proj.…` | 768 × 512 |
| `ref.vektor.0.…` / `ref.vektor.2.…` | `voice_encoder.vector_mlp.fc1.…` / `fc2.…` (SiLU between) | 768 × 512 / 768 × 768 |
| `dauer.stimme.…` | `duration_predictor.voice_proj.…` | 512 × 768 |
| `dauer.c1/c2/c3.…` | `duration_predictor.convs.0/1/2.…` | 384 × 512 × 3, 384 × 384 × 3 |
| `dauer.n1/n2/n3.…` | `duration_predictor.norms.0/1/2.…` | 384 |
| `dauer.aus.…` | `duration_predictor.out_proj.…` | 1 × 384 |
| `text_zu.…` | `dit.text_proj.…` | 384 × 512 |
| `tok_zu.…` | `dit.voice_token_proj.…` | 384 × 768 |
| `g_zu.…` | `dit.voice_vector_proj.…` | 384 × 768 |
| `ein.…` | `dit.input_proj.…` | 384 × 193 (columns: x_t 0–95, prompt 96–191, prompt flag 192) |
| `zeit.0.…` / `zeit.2.…` | `dit.time_mlp.fc1.…` / `fc2.…` (SiLU between) | 384 × 256 / 384 × 384 |
| `bloecke.{i}.n1.lin.…` | `dit.blocks.{i}.attn_norm.proj.…` (AdaNorm: [scale, shift]) | 768 × 384 |
| `bloecke.{i}.att.{q,kv,aus}.…` | `dit.blocks.{i}.attn.{q_proj,kv_proj,out_proj}.…` | |
| `bloecke.{i}.n2.…` | `dit.blocks.{i}.cross_norm.…` | 384 |
| `bloecke.{i}.kreuz.{q,kv,aus}.…` | `dit.blocks.{i}.cross_attn.{q_proj,kv_proj,out_proj}.…` | |
| `bloecke.{i}.n3.lin.…` | `dit.blocks.{i}.ff_norm.proj.…` | 768 × 384 |
| `bloecke.{i}.ff.{a,b}.…` | `dit.blocks.{i}.ff.{up,down}.…` | 1536 × 384 / 384 × 1536 |
| `norm.lin.…` | `dit.final_norm.proj.…` | 768 × 384 |
| `aus.…` | `dit.output_proj.…` | 96 × 384 |

### `codec.safetensors`

The training encoder keeps its layers in one sequence `encoder.stufen.0–13`: ConvNeXt blocks 0–3, stride-2
convolution 4, ConvNeXt 5–8, stride-2 convolution 9, ConvNeXt 10–13. The release numbers the blocks 0–11 in order.

| Training name | Release name | Shape |
|---|---|---|
| `encoder.ein.…` | `encoder.input_conv.…` | 512 × 100 × 7 |
| `encoder.stufen.{0–3, 5–8, 10–13}.*` | `encoder.blocks.{0–11}.*` (in this order) | |
| `encoder.stufen.4.…` / `encoder.stufen.9.…` | `encoder.downsample.0.…` / `encoder.downsample.1.…` | 512 × 512 × 4 |
| `encoder.norm.…` | `encoder.final_norm.…` (eps 1e-6) | 512 |
| `encoder.aus.…` (rows 0–95) | `encoder.output_proj.…` (mean only) | 96 × 512 |
| `decoder.ein.…` | `decoder.input_conv.…` | 512 × 96 × 3 |
| `decoder.hoch.…` | `decoder.upsample.…` (ConvTranspose1d k8 s4 p2) | 512 × 512 × 8 |
| `decoder.bloecke.{i}.*` | `decoder.blocks.{i}.*` | |
| `decoder.norm.…` | `decoder.final_norm.…` (eps 1e-6) | 512 |
| `decoder.kopf.…` | `decoder.head.…` (rows 0–480 log magnitude, 481–961 phase) | 962 × 512 |

ConvNeXt block (encoder and decoder): `dw.…` → `depthwise_conv.…` (512 × 1 × 7), `norm.…` → `norm.…` (eps 1e-6),
`up.…` → `up.…` (1536 × 512), `down.…` → `down.…` (512 × 1536), `gamma` → `scale` (512).

## `config.json`

| Key | Meaning |
|---|---|
| `sample_rate` | 24000 |
| `text.charset` | the 52 characters; id = index + 1, id 0 = padding / prompt placeholder |
| `text.languages` | `{"de": 0, "en": 1}` |
| `text_encoder`, `voice_encoder`, `duration_predictor`, `dit` | widths, layers, heads, FF factor |
| `latents.mean`, `latents.std` | 96 values each: `normalized = (mu - mean) / std`, and back before the decoder |
| `latents.frame_rate`, `latents.hop_length` | 25 Hz, 960 samples per latent frame |
| `codec.mel` | log-mel of the encoder (n_fft 1024, win 960, hop 240, 100 HTK bands, no norm, floor 1e-5) |
| `codec.istft` | inverse STFT of the decoder (n_fft 960, hop 240, periodic Hann, centered) |
| `sampler.timesteps` | [0, 0.25, 0.5, 0.75] |
| `sampler.prompt_frames` | 125: the first 5 s of the voice are the prompt the DiT continues from |
| `sampler.decoder_context_frames` | 12: prompt frames decoded in front of the new frames and cut off afterwards |
| `sampler.duration_units_per_frame` | 4: durations are in 10 ms units, 4 per latent frame |
| `duration_predictor.max_units` | 400: per-character duration cap |
| `voice.max_seconds` | 8: longer recordings are cut |
| `postprocess` | low-shelf +2 dB at 180 Hz (slope 0.7), peak limit 0.95 |
| `sentences` | at most 180 characters per piece, pieces with fewer than 20 letters are joined |
| `source` | training checkpoints and steps the release was exported from |
