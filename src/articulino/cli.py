"""Command line: ``articulino say "Text" --voice voice.wav -o out.wav`` and ``articulino normalize "Text"``."""
from __future__ import annotations

import argparse
import sys
import time


def _text(arg: str) -> str:
    return sys.stdin.read() if arg == "-" else arg


def main(argv: list[str] | None = None) -> int:
    from . import DEFAULT_REPO, __version__

    ap = argparse.ArgumentParser(prog="articulino", description="German and English text to speech with voice cloning.")
    ap.add_argument("--version", action="version", version=f"articulino {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    say = sub.add_parser("say", help="speak a text into a WAV file")
    say.add_argument("text", help='the text, or "-" to read it from standard input')
    say.add_argument("--voice", required=True, help="reference recording of the voice (3 to 8 s of clean speech)")
    say.add_argument("-o", "--output", default="speech.wav", help="WAV file to write (default: speech.wav)")
    say.add_argument("--model", default=DEFAULT_REPO, help="local folder or Hugging Face repository")
    say.add_argument("--device", default=None, help="cpu, cuda, ... (default: CUDA if available)")
    say.add_argument("--speed", type=float, default=1.0, help="speaking rate, 0.7 to 1.6 (default 1.0)")
    say.add_argument("--seed", type=int, default=0, help="the same seed gives the same audio (default 0)")
    say.add_argument("--language", choices=["de", "en"], default="de")
    say.add_argument("--no-normalize", action="store_true", help="skip the German text frontend")
    say.add_argument("--threads", type=int, default=None, help="CPU threads for PyTorch")

    norm = sub.add_parser("normalize", help="show what the German text frontend makes of a text")
    norm.add_argument("text", help='the text, or "-" to read it from standard input')

    a = ap.parse_args(argv)
    if a.command == "normalize":
        from .frontend import normalize

        print(normalize(_text(a.text)))
        return 0

    import torch

    from . import TTS

    if a.threads:
        torch.set_num_threads(a.threads)
    text = _text(a.text)
    tts = TTS.from_pretrained(a.model, device=a.device)
    tts.set_voice(a.voice)
    start = time.perf_counter()
    speech = tts.say(text, path=a.output, speed=a.speed, seed=a.seed, language=a.language,
                     normalize=not a.no_normalize)
    took = time.perf_counter() - start
    speedup = speech.duration / max(took, 1e-9)
    print(f"{a.output}: {speech.duration:.2f} s of audio in {took:.2f} s ({speedup:.1f}x real time)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
