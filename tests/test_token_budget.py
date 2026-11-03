from __future__ import annotations

import pytest

from literature_kb.parsing import ChunkRecipe, parse_document
from literature_kb.token_budget import split_to_budget


class CharTokenCounter:
    """Conservative fixture tokenizer; actual E5 audited separately against every chunk."""

    sha256 = "fixture-tokenizer"

    def count(self, title, text):
        return len(title) + len(text) + 5


def test_token_split_preserves_offsets_nonspace_coverage_and_budget(write_source):
    text = write_source()
    counter = CharTokenCounter()
    recipe = ChunkRecipe(180, 30, max_tokens=100, tokenizer_sha256=counter.sha256)
    parsed = parse_document(text, recipe, counter)
    covered = set()
    for chunk in parsed.chunks:
        assert counter.count(parsed.title, chunk.text) <= 100
        assert text[chunk.char_start : chunk.char_end] == chunk.text
        covered.update(range(chunk.char_start, chunk.char_end))
    for page in parsed.pages:
        assert all(
            page.char_start + i in covered for i, c in enumerate(page.text) if not c.isspace()
        )


def test_fitting_chunk_is_unchanged_and_wrong_counter_rejected():
    counter = CharTokenCounter()
    text = "short fixture"
    assert split_to_budget(text, "T", counter, 100, 20) == [(0, len(text))]
    with pytest.raises(ValueError):
        parse_document(
            "## Page 1\n" + "a" * 100,
            ChunkRecipe(max_tokens=100, tokenizer_sha256="wrong"),
            counter,
        )


def test_header_cannot_consume_entire_budget():
    from literature_kb.errors import KBError

    with pytest.raises(KBError, match="title"):
        split_to_budget("body", "t" * 100, CharTokenCounter(), 100, 20)


def test_legacy_recipe_fingerprint_is_preserved():
    import json

    from literature_kb.parsing import digest

    expected = digest(
        json.dumps(
            {
                "max_chars": 900,
                "overlap": 120,
                "min_page_chars": 30,
                "parser": "page-exact-offsets-v1",
            },
            sort_keys=True,
        )
    )
    assert ChunkRecipe().fingerprint == expected
