"""End-to-end listening check: speak the four sample sentences of the model card with say() (seed 0) and let a
speech recognizer write down what it hears.

The recognizer is any OpenAI-style /v1/audio/transcriptions endpoint (we used Parakeet-TDT 0.6B v3); 0.4 s of
silence is put in front of every clip, otherwise it tends to swallow the first sound.

  python tools/listen_check.py --asr http://127.0.0.1:8084 [--voice v.wav] [--out build/listen]

Comparison: lower case, letters and digits only, character error rate. The recognizer may write numbers as digits
where the frontend spelled them out, so a few points of CER can be formatting rather than pronunciation; read the
report. Writes the WAVs and a report.json to --out.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

SENTENCES = [  # (text, language) - the "Listen" table of the model card
    ("Guten Morgen! Dein Termin beim Zahnarzt ist heute um 15:30 Uhr in der Königstraße 12.", "de"),
    ("Die Stromrechnung über 87,40 € ist am 15.11. fällig; das PDF liegt in deinem Postfach.", "de"),
    ("Es gilt E = mc², und die Lichtgeschwindigkeit beträgt rund 300.000 km/s.", "de"),
    ("The delivery arrives tomorrow between ten and twelve.", "en"),
]
def transcribe(url: str, wav: np.ndarray, language: str, path: Path) -> str:
    import soundfile as sf

    padded = np.concatenate([np.zeros(int(0.4 * 24000), dtype=np.float32), wav])
    sf.write(path, padded, 24000, subtype="PCM_16")
    r = subprocess.run(["curl", "-s", f"{url.rstrip('/')}/v1/audio/transcriptions", "-F", f"file=@{path}",
                        "-F", f"language={language}"], capture_output=True, text=True, timeout=120)
    return json.loads(r.stdout).get("text", "")


def _simple(s: str) -> str:
    s = unicodedata.normalize("NFKC", s.lower())
    return re.sub(r"[^\w]", "", s)


def compare(heard: str, target_reading: str, language: str):
    """Character error rate between what was heard and what should have been said."""
    h, t = _simple(heard), _simple(target_reading)
    d = list(range(len(h) + 1))
    for i, c in enumerate(t, 1):
        p, d[0] = d[0], i
        for j, x in enumerate(h, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (c != x))
    return d[-1] / max(1, len(t))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", required=True, help="base URL of the speech recognizer")
    ap.add_argument("--model", default=str(HERE.parent / "build" / "release"))
    ap.add_argument("--voice", default=str(HERE.parent / "tests" / "data" / "radiov.wav"))
    ap.add_argument("--out", default=str(HERE.parent / "build" / "listen"))
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    import torch

    torch.set_num_threads(a.threads)
    from articulino import TTS
    from articulino.frontend import normalize

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tts = TTS.from_pretrained(a.model, device="cpu")
    tts.set_voice(a.voice)
    report = []
    for i, (text, lang) in enumerate(SENTENCES):
        speech = tts.say(text, language=lang, seed=0)
        reading = normalize(text) if lang == "de" else text
        heard = transcribe(a.asr, speech.audio, lang, out / f"{i}-{lang}.wav")
        cer = compare(heard, reading, lang)
        report.append({"text": text, "language": lang, "reading": reading, "heard": heard, "cer": round(cer, 3),
                       "seconds": round(speech.duration, 2)})
        print(f"[{i}] CER {cer:.3f} | {speech.duration:.2f} s\n    said:  {reading}\n    heard: {heard}", flush=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"mean CER {np.mean([r['cer'] for r in report]):.3f}; WAVs and report in {out}")


if __name__ == "__main__":
    main()
