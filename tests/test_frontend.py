"""The German text frontend against the expected readings of the 129-sentence pronunciation suite.

The expected readings in tests/data/pronunciation_cases.tsv are the output of the original frontend the model was
evaluated with; during development the port was also compared with that frontend on several thousand generated texts
and on random stream chunkings, without a single difference (see docs/DEVELOPMENT.md)."""
from __future__ import annotations

import pytest

from articulino.frontend import normalize
from conftest import HERE

CASES = HERE / "data" / "pronunciation_cases.tsv"


def _cases():
    rows = []
    for line in CASES.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            rows.append(tuple(line.split("\t")))
    return rows


def test_pronunciation_suite_matches_reference_output():
    """All 129 sentences of the pronunciation suite come out exactly as from the original frontend."""
    rows = _cases()
    assert len(rows) == 129
    wrong = [(inp, ref, normalize(inp)) for _, inp, _, ref in rows if normalize(inp) != ref]
    assert not wrong, wrong[:3]
