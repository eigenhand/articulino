"""Export the trained checkpoints to the release format of the `articulino` package.

Reads the two training checkpoints (never modified):

* acoustic model (student after DMD2): ``schueler-dmd1-020000.pt`` -> ``model.safetensors``
* latent codec (encoder + fine-tuned decoder d1): ``vocoder-d1-020000.pt`` -> ``codec.safetensors``

and writes ``config.json`` with every size, the latent statistics, the sampling times, the sample rate and the
character set. Tensor names are rewritten from the German training names to clean English names; the full
old -> new table is written to ``tensor_mapping.json`` next to the weights and documented in ``docs/weights.md``.

Dropped on purpose (training-only, never used by the 4-step student at inference):

* ``null_text``, ``ref.null_tokens``, ``ref.null_vektor``: "no text" / "no voice" embeddings for classifier-free
  guidance during training.
* the ``logvar`` half of ``encoder.aus`` (rows 96..191): the encoder's variance head (VAE training only).
* ``encoder.mel.*`` buffers (Hann window and mel filter bank): recomputed exactly in code.

Usage::

    python tools/export_safetensors.py --acoustic schueler-dmd1-020000.pt --codec vocoder-d1-020000.pt [--out build/release]

The training checkpoints are not public; the released files are the output of this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / "build" / "release"

CHARSET = " abcdefghijklmnopqrstuvwxyzäöüß.,?!-'\":;()0123456789"

# --------------------------------------------------------------------------------------------- name rules
# (pattern on the old name, replacement); the first matching rule wins. Every old tensor must match exactly one rule
# or be listed in DROPPED_*; the export fails otherwise.
_BLOCK = [  # pre-norm transformer block of the text and voice encoders
    (r"n1\.(weight|bias)", r"attn_norm.\1"),
    (r"n2\.(weight|bias)", r"ff_norm.\1"),
    (r"att\.q\.(weight|bias)", r"attn.q_proj.\1"),
    (r"att\.kv\.(weight|bias)", r"attn.kv_proj.\1"),
    (r"att\.aus\.(weight|bias)", r"attn.out_proj.\1"),
    (r"ff\.a\.(weight|bias)", r"ff.up.\1"),
    (r"ff\.b\.(weight|bias)", r"ff.down.\1"),
]


def _shift(replacement: str) -> str:
    """Group 1 of a nested rule is the layer index, so the inner rule's group 1 becomes group 2."""
    return replacement.replace(r"\1", r"\2")


def _block_rules(old_prefix: str, new_prefix: str) -> list[tuple[str, str]]:
    return [(rf"{old_prefix}\.(\d+)\.{p}", rf"{new_prefix}.\1.{_shift(r)}") for p, r in _BLOCK]


ACOUSTIC_RULES: list[tuple[str, str]] = [
    # text encoder
    (r"text\.emb\.weight", "text_encoder.char_embedding.weight"),
    (r"text\.sprache\.weight", "text_encoder.language_embedding.weight"),
    *_block_rules(r"text\.bloecke", "text_encoder.layers"),
    (r"text\.norm\.(weight|bias)", r"text_encoder.final_norm.\1"),
    # voice encoder
    (r"ref\.ein\.(weight|bias)", r"voice_encoder.input_proj.\1"),
    *_block_rules(r"ref\.bloecke", "voice_encoder.layers"),
    (r"ref\.fragen", "voice_encoder.queries"),
    (r"ref\.pool\.(\d+)\.0\.(weight|bias)", r"voice_encoder.pool.\1.attn_norm.\2"),
    (r"ref\.pool\.(\d+)\.1\.q\.(weight|bias)", r"voice_encoder.pool.\1.attn.q_proj.\2"),
    (r"ref\.pool\.(\d+)\.1\.kv\.(weight|bias)", r"voice_encoder.pool.\1.attn.kv_proj.\2"),
    (r"ref\.pool\.(\d+)\.1\.aus\.(weight|bias)", r"voice_encoder.pool.\1.attn.out_proj.\2"),
    (r"ref\.pool\.(\d+)\.2\.(weight|bias)", r"voice_encoder.pool.\1.ff_norm.\2"),
    (r"ref\.pool\.(\d+)\.3\.a\.(weight|bias)", r"voice_encoder.pool.\1.ff.up.\2"),
    (r"ref\.pool\.(\d+)\.3\.b\.(weight|bias)", r"voice_encoder.pool.\1.ff.down.\2"),
    (r"ref\.norm\.(weight|bias)", r"voice_encoder.final_norm.\1"),
    (r"ref\.aus\.(weight|bias)", r"voice_encoder.token_proj.\1"),
    (r"ref\.vektor\.0\.(weight|bias)", r"voice_encoder.vector_mlp.fc1.\1"),
    (r"ref\.vektor\.2\.(weight|bias)", r"voice_encoder.vector_mlp.fc2.\1"),
    # duration predictor
    (r"dauer\.stimme\.(weight|bias)", r"duration_predictor.voice_proj.\1"),
    (r"dauer\.c1\.(weight|bias)", r"duration_predictor.convs.0.\1"),
    (r"dauer\.c2\.(weight|bias)", r"duration_predictor.convs.1.\1"),
    (r"dauer\.c3\.(weight|bias)", r"duration_predictor.convs.2.\1"),
    (r"dauer\.n1\.(weight|bias)", r"duration_predictor.norms.0.\1"),
    (r"dauer\.n2\.(weight|bias)", r"duration_predictor.norms.1.\1"),
    (r"dauer\.n3\.(weight|bias)", r"duration_predictor.norms.2.\1"),
    (r"dauer\.aus\.(weight|bias)", r"duration_predictor.out_proj.\1"),
    # DiT (student)
    (r"text_zu\.(weight|bias)", r"dit.text_proj.\1"),
    (r"tok_zu\.(weight|bias)", r"dit.voice_token_proj.\1"),
    (r"g_zu\.(weight|bias)", r"dit.voice_vector_proj.\1"),
    (r"ein\.(weight|bias)", r"dit.input_proj.\1"),
    (r"zeit\.0\.(weight|bias)", r"dit.time_mlp.fc1.\1"),
    (r"zeit\.2\.(weight|bias)", r"dit.time_mlp.fc2.\1"),
    (r"bloecke\.(\d+)\.n1\.lin\.(weight|bias)", r"dit.blocks.\1.attn_norm.proj.\2"),
    (r"bloecke\.(\d+)\.n2\.(weight|bias)", r"dit.blocks.\1.cross_norm.\2"),
    (r"bloecke\.(\d+)\.n3\.lin\.(weight|bias)", r"dit.blocks.\1.ff_norm.proj.\2"),
    (r"bloecke\.(\d+)\.att\.q\.(weight|bias)", r"dit.blocks.\1.attn.q_proj.\2"),
    (r"bloecke\.(\d+)\.att\.kv\.(weight|bias)", r"dit.blocks.\1.attn.kv_proj.\2"),
    (r"bloecke\.(\d+)\.att\.aus\.(weight|bias)", r"dit.blocks.\1.attn.out_proj.\2"),
    (r"bloecke\.(\d+)\.kreuz\.q\.(weight|bias)", r"dit.blocks.\1.cross_attn.q_proj.\2"),
    (r"bloecke\.(\d+)\.kreuz\.kv\.(weight|bias)", r"dit.blocks.\1.cross_attn.kv_proj.\2"),
    (r"bloecke\.(\d+)\.kreuz\.aus\.(weight|bias)", r"dit.blocks.\1.cross_attn.out_proj.\2"),
    (r"bloecke\.(\d+)\.ff\.a\.(weight|bias)", r"dit.blocks.\1.ff.up.\2"),
    (r"bloecke\.(\d+)\.ff\.b\.(weight|bias)", r"dit.blocks.\1.ff.down.\2"),
    (r"norm\.lin\.(weight|bias)", r"dit.final_norm.proj.\1"),
    (r"aus\.(weight|bias)", r"dit.output_proj.\1"),
]
DROPPED_ACOUSTIC = {"null_text", "ref.null_tokens", "ref.null_vektor"}

# encoder.stufen is one Sequential: ConvNeXt 0-3, down-sampling conv 4, ConvNeXt 5-8, down-sampling conv 9,
# ConvNeXt 10-13. In the release: encoder.blocks.0-11 (in order) and encoder.downsample.0-1.
_ENC_BLOCK = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 6: 5, 7: 6, 8: 7, 10: 8, 11: 9, 12: 10, 13: 11}
_ENC_DOWN = {4: 0, 9: 1}
_CONVNEXT = [(r"dw\.(weight|bias)", r"depthwise_conv.\1"), (r"norm\.(weight|bias)", r"norm.\1"),
             (r"up\.(weight|bias)", r"up.\1"), (r"down\.(weight|bias)", r"down.\1"), (r"gamma", "scale")]

CODEC_RULES: list[tuple[str, str]] = [
    (r"encoder\.ein\.(weight|bias)", r"encoder.input_conv.\1"),
    *[(rf"encoder\.stufen\.{o}\.{p}", rf"encoder.blocks.{n}.{r}") for o, n in _ENC_BLOCK.items() for p, r in _CONVNEXT],
    *[(rf"encoder\.stufen\.{o}\.(weight|bias)", rf"encoder.downsample.{n}.\1") for o, n in _ENC_DOWN.items()],
    (r"encoder\.norm\.(weight|bias)", r"encoder.final_norm.\1"),
    (r"encoder\.aus\.(weight|bias)", r"encoder.output_proj.\1"),       # sliced to the mu half below
    (r"decoder\.ein\.(weight|bias)", r"decoder.input_conv.\1"),
    (r"decoder\.hoch\.(weight|bias)", r"decoder.upsample.\1"),
    *[(rf"decoder\.bloecke\.(\d+)\.{p}", rf"decoder.blocks.\1.{_shift(r)}") for p, r in _CONVNEXT],
    (r"decoder\.norm\.(weight|bias)", r"decoder.final_norm.\1"),
    (r"decoder\.kopf\.(weight|bias)", r"decoder.head.\1"),
]
DROPPED_CODEC = {"encoder.mel.mel.spectrogram.window", "encoder.mel.mel.mel_scale.fb"}


def rename(state: dict[str, torch.Tensor], rules: list[tuple[str, str]], dropped: set[str]) -> tuple[OrderedDict, dict]:
    """Apply the rules to every tensor name. Returns (new state dict, old -> new mapping)."""
    out, mapping = OrderedDict(), {}
    for old, tensor in state.items():
        if old in dropped:
            mapping[old] = None
            continue
        hits = [re.sub(rf"^{p}$", r, old) for p, r in rules if re.fullmatch(p, old)]
        if len(hits) != 1:
            raise KeyError(f"{len(hits)} rules match tensor {old!r}")
        new = hits[0]
        if new in out:
            raise KeyError(f"two tensors map to {new!r}")
        out[new] = tensor.detach().to(torch.float32).contiguous().clone()
        mapping[old] = new
    return out, mapping


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def build_config(ck: dict, vck: dict) -> dict:
    cfg, vcfg = ck["cfg"], vck["cfg"]
    assert cfg["v"] == len(CHARSET) + 1 and cfg.get("anfang_eigen") and cfg.get("lehrer_dim") == 768, cfg
    mean = torch.as_tensor(ck["mittel"]).float()
    std = torch.as_tensor(ck["streuung"]).float()
    return {
        "model_type": "articulino",
        "format_version": 1,
        "sample_rate": 24000,
        "text": {
            "charset": CHARSET,
            "pad_id": 0,
            "vocab_size": cfg["v"],
            "languages": {"de": 0, "en": 1},
        },
        "text_encoder": {"dim": cfg["text_dim"], "layers": cfg["text_lagen"], "heads": 8, "ff_mult": 4,
                         "layer_norm_eps": 1e-5},
        "voice_encoder": {"input_dim": cfg["latent"], "dim": cfg["text_dim"], "layers": 4, "heads": 8,
                          "num_tokens": 32, "out_dim": cfg["lehrer_dim"], "ff_mult": 4, "pool_layers": 2},
        "duration_predictor": {"input_dim": cfg["text_dim"], "voice_dim": cfg["lehrer_dim"], "channels": 384,
                               "kernel_size": 3, "max_units": 400.0, "unit_ms": 10},
        "dit": {"dim": cfg["dim"], "layers": cfg["lagen"], "heads": cfg["koepfe"], "ff_mult": 4,
                "latent_dim": cfg["latent"], "text_dim": cfg["text_dim"], "voice_dim": cfg["lehrer_dim"],
                "time_embedding_dim": 256, "time_scale": 1000.0, "rope_base": 10000.0},
        "latents": {"dim": cfg["latent"], "frame_rate": 25, "hop_length": 960,
                    "mean": [float(x) for x in mean], "std": [float(x) for x in std]},
        "codec": {
            "mel": {"n_fft": 1024, "win_length": 960, "hop_length": 240, "n_mels": 100, "f_min": 0.0,
                    "f_max": 12000.0, "power": 1.0, "center": True, "pad_mode": "reflect", "mel_scale": "htk",
                    "norm": None, "log_floor": 1e-5},
            "encoder": {"dim": vcfg["enc_dim"], "blocks_per_stage": 4, "stages": 3, "ff_mult": 3,
                        "kernel_size": 7, "layer_norm_eps": 1e-6, "latent_dim": vcfg["latent"]},
            "decoder": {"dim": vcfg["dec_dim"], "blocks": vcfg["dec_bloecke"], "ff_mult": 3, "kernel_size": 7,
                        "upsample": 4, "layer_norm_eps": 1e-6, "max_magnitude": 100.0, "latent_dim": vcfg["latent"]},
            "istft": {"n_fft": 960, "hop_length": 240, "win_length": 960, "window": "hann_periodic", "center": True},
        },
        "sampler": {
            "timesteps": [float(t) for t in ck.get("zeiten", (0.0, 0.25, 0.5, 0.75))],
            "prompt_frames": 125,
            "decoder_context_frames": 12,
            "duration_units_per_frame": 4,
        },
        "voice": {"max_seconds": 8.0},
        "postprocess": {"low_shelf_db": 2.0, "low_shelf_hz": 180.0, "shelf_slope": 0.7, "peak_limit": 0.95},
        "sentences": {"max_chars": 180, "min_letters": 20},
        "source": {"acoustic_checkpoint": "schueler-dmd1-020000.pt", "acoustic_step": int(ck.get("schritt", -1)),
                   "codec_checkpoint": "vocoder-d1-020000.pt", "codec_step": int(vck.get("schritt", -1))},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--acoustic", type=Path, required=True, help="student checkpoint (.pt)")
    ap.add_argument("--codec", type=Path, required=True, help="codec checkpoint (.pt)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()

    ck = torch.load(a.acoustic, map_location="cpu", weights_only=False)
    vck = torch.load(a.codec, map_location="cpu", weights_only=False)

    model, map_model = rename(ck["ema"], ACOUSTIC_RULES, DROPPED_ACOUSTIC)
    codec, map_codec = rename(vck["ae"], CODEC_RULES, DROPPED_CODEC)
    latent = vck["cfg"]["latent"]
    for k in ("encoder.output_proj.weight", "encoder.output_proj.bias"):   # keep mu, drop logvar
        codec[k] = codec[k][:latent].contiguous().clone()

    config = build_config(ck, vck)
    a.out.mkdir(parents=True, exist_ok=True)
    meta = {"format": "pt", "model_type": "articulino"}
    save_file(model, a.out / "model.safetensors", metadata={**meta, "part": "acoustic"})
    save_file(codec, a.out / "codec.safetensors", metadata={**meta, "part": "codec"})
    (a.out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (a.out / "tensor_mapping.json").write_text(json.dumps({"model.safetensors": map_model,
                                                           "codec.safetensors": map_codec},
                                                          ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    n_model = sum(t.numel() for t in model.values())
    n_codec = sum(t.numel() for t in codec.values())
    print(f"model.safetensors  {len(model):4d} tensors  {n_model / 1e6:7.2f} M  sha256 {sha256(a.out / 'model.safetensors')[:16]}")
    print(f"codec.safetensors  {len(codec):4d} tensors  {n_codec / 1e6:7.2f} M  sha256 {sha256(a.out / 'codec.safetensors')[:16]}")
    print(f"dropped: {sorted(k for k, v in {**map_model, **map_codec}.items() if v is None)} + encoder logvar rows")
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
