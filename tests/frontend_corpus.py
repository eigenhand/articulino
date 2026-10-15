"""Generated German inputs for frontend tests (deterministic, no files needed)."""
from __future__ import annotations

import json
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "src" / "articulino" / "frontend" / "data"

FIXED = [
    "Guten Morgen! Dein Termin beim Zahnarzt ist heute um 15:30 Uhr in der Königstraße 12.",
    "Die Stromrechnung über 87,40 € ist am 15.11. fällig; das PDF liegt in deinem Postfach.",
    "Es gilt E = mc², und die Lichtgeschwindigkeit beträgt rund 300.000 km/s.",
    "The delivery arrives tomorrow between ten and twelve.",
    "Guten Morgen! Der Termin ist um 15:30 Uhr.",
    "Das [en]Meeting[/en] ist verschoben.",
    "## Überblick\n\nDeutschland hat **drei** Stärken:\n\n1. Industrie\n2. *Forschung*\n3. Lage\n\n"
    "| Stadt | Einwohner |\n|---|---|\n| Berlin | 3,7 Mio. |\n\n`code` und [Link](https://example.org)\n\n```\nprint(1)\n```",
    "Die Lösung ist $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$ und $\\int_0^1 x^2 dx = \\frac{1}{3}$.",
    "Schreib an max.mustermann@example.de oder besuche https://www.eigenhand.dev/srv/abc?x=1 bitte.",
    "Ruf an: +49 170 1234567 oder 0911/12345-6, PLZ 90402, in 90402 Nürnberg.",
    "Mo–Fr 9-17 Uhr, Sa. 10–14 Uhr. So geschlossen.",
    "Im 3. Jh. v. Chr. Er zeigte z. B. u. a. 2x täglich, 3x + 2 = 8.",
    "Am Freitag, 25. Juli 2025 und am 28.11. sowie 05.12. war es kalt: -5 °C bis 3 °C.",
    "Das kostet 1.250,50 € bzw. ca. 19,99 $ oder $5 inkl. MwSt.",
    "BITTE SOFORT MELDEN! Die GPU, BITTE prüfen. DAS IST NICHT GUT. Ludwig II und Plan B.",
    "Die Datei liegt in /srv/daten und heißt neu_bauen.py; web_ui läuft.",
    "Wir haben 24/7 offen, 3/4 sind voll, 9/11 war schlimm, 1/2 Liter.",
    "Version 3.2.1 und IP 192.168.0.1, Python 3.12. ab 2026.",
    "Super 😂 toll! Das regnet :) Nimm ¯\\_(ツ)_/¯ Berlin → Hauptstadt, A -> B.",
    "f(x) = 2x - 1, a² + b² = c², √2 ≈ 1,41, 10⁻³, H₂O, 2^10, 5 * 3, 12 / 4.",
    "Was ergibt 10 - 4? 10 - 4 ergibt 6. Es sind 20–25 °C, also 22 ± 3 Grad.",
    "> Zitat hier\n- Punkt eins\n* Punkt zwei\n+ Punkt drei\n---\n### Titel:\nText",
]

WORDS = ["Der", "die", "das", "am", "im", "vom", "zum", "bis", "und", "oder", "zwischen", "ab", "seit", "Er", "Sie",
         "Heute", "Montag", "Freitag,", "um", "in", "mit", "für", "über", "auf", "ein", "eine", "einen", "einer",
         "Termin", "Treffen", "Preis", "Länge", "Strom", "Haus", "Auto", "Plan", "Version", "Seite", "Kapitel",
         "Nummer", "PDF", "KI", "GPU", "USB", "ADAC", "NASA", "NATO", "TÜV", "RAMs", "PINs", "IT", "CEO", "LKW", "srv",
         "mnt", "hmm", "x", "y", "a", "b", "E", "I", "C", "BITTE", "SOFORT", "NICHT", "ACHTUNG", "OK", "okay", "ok",
         "nginx", "JSON", "iOS", "macOS", "GmbH", "C++", "Python", "daten.py", "neu_bauen", "**fett**", "*kursiv*",
         "`code`", "[Link](http://a.de)", "😀", ":)", ";-)", "<3", "->", "→", "Januar", "März", "Mai", "Dezember",
         "Uhr", "Euro", "€", "$", "%", "°C", "°", "§", "&", "=", "±", "≈", "≤", "≥", "+", "-", "–", "−", "×", "·", "/",
         "÷", "^", "√", "π", "∞", "²", "³", "₂", "(", ")", "#", "@", "z. B.", "d. h.", "u. a.", "bzw.", "usw.",
         "ca.", "Nr.", "Dr.", "Str.", "Mio.", "Mrd.", "Mo.", "Di.", "v. Chr.", "n. Chr.", "Jh.", "max.", "Tel.",
         "Mo–Fr", "Sa-So", "PLZ", "Postleitzahl", "ergibt", "ist", "gleich", "plus", "mal", "x²", "mc²", "cm²", "m²"]
NUMBERS = ["0", "1", "2", "3", "7", "12", "15", "24", "31", "42", "99", "100", "101", "1000", "1.000", "1.250",
           "2.500.000", "1990", "2026", "1100", "1999", "0911", "123456", "1234567890", "1,5", "1,82", "19,99",
           "3,14159", "87,40", "1.250,50", "15:30", "9:05", "24:00", "2:1", "3.", "25.", "1.", "29.09.2026", "3.10.",
           "28.11.", "05.12.", "1.1.", "13.13.", "3/4", "1/2", "24/7", "9/11", "50-60", "7-9", "1990-1995", "8.000–12.000",
           "3.2.1", "192.168.0.1", "#1", "2x", "4×", "-5", "+49", "+49 170 1234567", "0911 123456", "0911/12345-6",
           "90402", "10⁻³", "2^10", "1e5", "10 - 4", "5 + 3", "5 * 3", "12 / 4", "3x", "2a"]
SEPARATORS = [" ", " ", " ", ", ", ". ", "! ", "? ", ": ", "; ", " - ", "\n", "\n\n", "", " (", ") ", " „", "“ "]


def _units() -> list[str]:
    return list(json.loads((DATA / "units.json").read_text(encoding="utf-8"))["units"])


def _abbreviations() -> list[str]:
    d = json.loads((DATA / "abbreviations.json").read_text(encoding="utf-8"))
    return list(d["abbreviations"]) + list(d["symbols"])


def corpus(n_random: int = 3000, seed: int = 2026) -> list[str]:
    """Fixed sentences, every unit and abbreviation in context, and random token soups."""
    out = list(FIXED)
    for u in _units():
        out += [f"Es sind 1 {u} gemessen.", f"Wir brauchen 2{u} heute.", f"Mit 1,5 {u}/h und 3-4 {u}.",
                f"Etwa -1 {u} für 1,80 {u} in 10 {u}."]
    for k in _abbreviations():
        out += [f"Das ist {k} wichtig.", f"Siehe {k}", f"Es war {k} Er kam später.", f"{k}"]
    rng = random.Random(seed)
    pool = WORDS + NUMBERS + _units()[:60]
    for _ in range(n_random):
        k = rng.randint(1, 14)
        parts = []
        for _ in range(k):
            parts.append(rng.choice(pool) if rng.random() < 0.8 else rng.choice(NUMBERS))
            parts.append(rng.choice(SEPARATORS))
        out.append("".join(parts).strip() if rng.random() < 0.7 else "".join(parts))
    return out


def chunkings(text: str, rng: random.Random, n: int = 3) -> list[list[str]]:
    """Random ways to cut a text into stream pieces."""
    out = []
    for _ in range(n):
        pieces, i = [], 0
        while i < len(text):
            j = min(len(text), i + rng.randint(1, 12))
            pieces.append(text[i:j])
            i = j
        out.append(pieces)
    return out
