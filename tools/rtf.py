"""Real-time factor on the CPU, measured like the original evaluation script.

Ten sentences, one voice, the voice cache built beforehand, one warm-up sentence, then every sentence through the
whole chain (text -> audio incl. post-processing), seed 42 + i. Prints audio seconds per compute second and the time
until the first sentence is ready.

  python tools/rtf.py --threads 12 [--model build/release] [--voice v.wav] [--sentences s.txt]
      [--mode api|raw]

--mode api: tts.say() with the defaults (German frontend, sentence splitting, uniform rate, low-shelf).
--mode raw: the same work as the reference script (no frontend, no uniform rate), for a like-for-like comparison.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
SENTENCES = (  # the ten reference sentences of every speed measurement
    "Guten Morgen, die Systeme laufen stabil und es gibt nichts Dringendes.",
    "Ich habe die Unterlagen geprüft und fasse das Wichtigste kurz zusammen.",
    "Der Termin am Donnerstag wurde auf den Nachmittag verschoben.",
    "Soll ich die Nachricht jetzt abschicken oder noch einmal vorlesen?",
    "Draußen sind es zwölf Grad, und am Abend soll es regnen.",
    "Die Lieferung kommt voraussichtlich morgen zwischen zehn und zwölf Uhr.",
    "Ich habe drei neue Mails gefunden, eine davon ist von deiner Bank.",
    "Das klingt nach einem guten Plan, ich kümmere mich gleich darum.",
    "Kurz zur Erinnerung: Du wolltest heute noch den Drucker reinigen.",
    "Alles erledigt. Gibt es sonst noch etwas, das ich für dich tun kann?",
)

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE.parent / "build" / "release"))
    ap.add_argument("--voice", default=str(HERE.parent / "tests" / "data" / "radiov.wav"))
    ap.add_argument("--sentences", default=None, help="one sentence per line (default: the ten reference sentences)")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--mode", choices=["api", "raw"], default="api")
    ap.add_argument("--repeat", type=int, default=1, help="repeat the ten sentences and keep the best run")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)

    from articulino import TTS
    from articulino.text import encode

    sentences = ([z.strip() for z in open(a.sentences, encoding="utf-8") if z.strip()] if a.sentences
                 else list(SENTENCES))
    tts = TTS.from_pretrained(a.model, device="cpu", uniform_rate=(a.mode == "api"))
    tts.set_voice(a.voice)

    def speak(text: str, seed: int) -> np.ndarray:
        if a.mode == "api":
            return tts.say(text, seed=seed).audio
        ids, langs = encode(text)
        return tts._postprocess(*tts.synthesize(ids, langs, 1.0, seed))

    speak("Ein kurzer Satz zum Aufwärmen.", 0)
    best = None
    for _ in range(a.repeat):
        times, seconds, first = [], 0.0, None
        for i, s in enumerate(sentences):
            t0 = time.perf_counter()
            w = speak(s, 42 + i)
            dt = time.perf_counter() - t0
            times.append(dt)
            seconds += len(w) / 24000
            first = dt if first is None else first
        if best is None or sum(times) < sum(best[0]):
            best = (times, seconds, first)
    times, seconds, first = best
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    print(f"articulino {a.mode} | CPUs {cpus[0]}-{cpus[-1]} ({len(cpus)}) | {a.threads} threads | {seconds:.1f} s audio in "
          f"{sum(times):.2f} s = {seconds / sum(times):.1f}x real time | first sentence {first:.2f} s, median per "
          f"sentence {np.median(times):.2f} s | torch {torch.__version__}", flush=True)


if __name__ == "__main__":
    main()
