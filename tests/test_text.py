"""Characters, language tags, sentence splitting and post-processing."""
from __future__ import annotations

import random

import numpy as np
import pytest

from articulino.postprocess import SosFilter, limit_peak, shelf_sos
from articulino.sentences import SentenceBuffer, carry_language_tags, split_sentences
from articulino.text import CHARSET, VOCAB_SIZE, encode, normalize_characters, text_to_ids
def test_charset():
    assert VOCAB_SIZE == 53 and len(CHARSET) == 52
    assert normalize_characters("Café “Olé” – naïve Ærø…") == 'cafe "ole" - naive ärö.'
    assert text_to_ids("Ab") == [2, 3]


def test_language_tags():
    ids, langs = encode("Das [en]Meeting[/en] ist verschoben.")
    text = "".join(CHARSET[i - 1] for i in ids)
    assert text == "das meeting ist verschoben."
    assert langs == [0] * 4 + [1] * 7 + [0] * (len(ids) - 11)
    assert encode("Hello there", "en")[1] == [1] * 11
    with pytest.raises(ValueError):
        encode("x", "fr")


def test_carry_language_tags():
    assert carry_language_tags(["Er sagt [en]hello.", "How are you?[/en] Gut."]) == \
        ["Er sagt [en]hello.", "[en]How are you?[/en] Gut."]


def test_sentence_buffer_matches_one_shot():
    text = ("Guten Morgen! Heute regnet es ab achtzehn Uhr. Nimm einen Schirm mit. Danach wird es kalt, sehr kalt, "
            "und morgen früh ist es glatt, also fahr bitte vorsichtig, nimm dir Zeit, und ruf an, wenn du da bist, "
            "denn ich mache mir sonst Sorgen, wie immer eben, du kennst mich ja.")
    one = split_sentences(text, True)[0]
    rng = random.Random(5)
    for _ in range(50):
        buf, out, i = SentenceBuffer(), [], 0
        while i < len(text):
            j = min(len(text), i + rng.randint(1, 15))
            out += buf.push(text[i:j])
            i = j
        out += buf.flush()
        assert " ".join(out) == " ".join(one)
    assert all(len(s) <= 180 for s in one) and len(one) >= 3


def test_filter_like_scipy():
    signal = pytest.importorskip("scipy.signal")
    x = np.random.default_rng(0).standard_normal(50_000).astype(np.float32) * 0.3
    for sos in (np.stack([shelf_sos(180, 2.0, False), shelf_sos(1000, 0.0, True)]), shelf_sos(3000, -6, True)[None]):
        f = SosFilter(sos)
        assert f(x[:0]).shape == (0,)
        for n in (1, 7, 1000, 1231, 50_000):
            assert np.abs(f(x[:n]) - signal.sosfilt(sos, x[:n])).max() < 1e-11


def test_limit_peak():
    x = np.array([0.5, -1.9, 0.2], dtype=np.float32)
    y = limit_peak(x, 0.95)
    assert y.dtype == np.float32 and abs(np.abs(y).max() - 0.95) < 1e-7
    assert limit_peak(x * 0.1, 0.95) is not None and np.array_equal(limit_peak(x * 0.1, 0.95), x * 0.1)
    assert limit_peak(x, None) is x


def test_foreign_spans_skip_the_german_frontend():
    from articulino.tts import _TaggedStreamNormalizer, _normalize_german

    assert _normalize_german("Das [en]PDF[/en] und [en]a[/en] Buch, z. B. am 3.10.") == \
        "Das [en]PDF[/en] und [en]a[/en] Buch, zum Beispiel am dritten Oktober."
    assert _normalize_german("Am 3.10. um 9:05 Uhr.") == "Am dritten Oktober um neun Uhr fünf."
    text = "Er sagte: [en]I am fine, thank you. A great day for a walk.[/en] Danach ging er um 15:30 Uhr nach Hause."
    s = _TaggedStreamNormalizer()
    out = "".join([s.push(text[i:i + 5]) for i in range(0, len(text), 5)] + [s.flush()])
    assert "[en]I am fine, thank you." in out and "A great day for a walk.[/en]" in out
    assert "fünfzehn Uhr dreißig" in out


def test_foreign_spans_always_restored():
    from articulino import frontend
    from articulino.text import protect_foreign_spans, restore_foreign_spans
    from frontend_corpus import corpus

    rng = random.Random(11)
    for x in corpus(1500):
        words = x.split(" ")
        if len(words) < 2:
            continue
        i = rng.randrange(len(words))
        j = min(len(words), i + rng.randint(1, 3))
        tagged = " ".join(words[:i] + ["[en]" + " ".join(words[i:j]) + ("[/en]" if rng.random() < 0.8 else "")]
                          + words[j:])
        masked, spans = protect_foreign_spans(tagged)
        out = restore_foreign_spans(frontend.normalize(masked), spans)
        assert out is not None and all(span in out for span in spans), tagged


def test_space_between_tags():
    ids, langs = encode("[en]Item one[/en] [en]Item two[/en]")
    assert "".join(CHARSET[i - 1] for i in ids) == "item one item two" and langs == [1] * 8 + [0] + [1] * 8
