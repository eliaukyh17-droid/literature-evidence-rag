from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from literature_kb.errors import KBError
from literature_kb.evaluation import evaluate
from literature_kb.lexical import query_expression, tokens
from literature_kb.parsing import ChunkRecipe, parse_document
from literature_kb.store import Store


def test_raw_offsets_roundtrip_and_overlap(write_source):
    text = write_source()
    parsed = parse_document(text, ChunkRecipe(180, 30))
    assert len(parsed.chunks) > 4
    for chunk in parsed.chunks:
        page = parsed.pages[chunk.page - 1]
        assert text[chunk.char_start : chunk.char_end] == chunk.text
        assert page.text[chunk.page_char_start : chunk.page_char_end] == chunk.text
        assert len(chunk.text) <= 180
    for page in parsed.pages:
        parts = [c for c in parsed.chunks if c.page == page.number]
        assert all(a.char_end >= b.char_start for a, b in zip(parts, parts[1:], strict=False))


@pytest.mark.parametrize("text", ["no markers", "## Page 2\nx", "## Page 1\nx\n## Page 1\ny"])
def test_bad_page_structure_rejected(text):
    with pytest.raises(KBError, match="sequential"):
        parse_document(text, ChunkRecipe())


def test_ocr_unavailable_and_short_pages_excluded():
    text = (
        "## Page 1\n[OCR_REQUIRED]\n" + "bad" * 100 + "\n## Page 2\nx\n## Page 3\n" + "good " * 100
    )
    parsed = parse_document(text, ChunkRecipe())
    assert parsed.skipped_pages == [1, 2]
    assert all(c.page == 3 for c in parsed.chunks)


def test_invalid_recipe():
    with pytest.raises(ValueError):
        ChunkRecipe(100, 80)


@pytest.mark.parametrize(
    "path",
    ["../secret.md", "C:\\keys.md", "/secret.md", "test.md:stream", "test.csv", "a/../../b.md"],
)
def test_path_scope(store, path):
    with pytest.raises(KBError) as err:
        store.import_document(path)
    assert err.value.code == "invalid_source"


def test_oversize_and_bad_encoding(settings, source):
    from dataclasses import replace

    limited = Store(replace(settings, max_source_bytes=5))
    (source / "x.md").write_bytes(b"a" * 6)
    with pytest.raises(KBError) as err:
        limited.import_document("x.md")
    assert err.value.status == 413
    (source / "x.md").write_bytes(b"\xff\xff")
    with pytest.raises(KBError) as err:
        limited.import_document("x.md")
    assert err.value.code == "invalid_encoding"


def test_import_is_idempotent_duplicate_dedup_and_restart(store, settings, write_source):
    text = write_source()
    first = store.import_document("test.md")
    assert first["status"] == "imported"
    assert store.import_document("test.md")["status"] == "unchanged"
    write_source("copy.md", text)
    assert store.import_document("copy.md")["status"] == "duplicate"
    restarted = Store(settings)
    assert restarted.stats()["documents"] == 1
    assert restarted.search("证据定位")["hits"]


def test_update_atomic_visibility_revert_and_delete(store, write_source):
    original = write_source()
    first = store.import_document("test.md")
    old_hit = store.search("证据定位")["hits"][0]
    write_source(body="## Page 5\nbad update")
    with pytest.raises(KBError):
        store.import_document("test.md")
    assert store.evidence(old_hit["chunk_id"])["text"] == old_hit["text"]
    write_source(body="# New\n## Page 1\n" + "新的唯一词彗星望远镜和索引更新。" * 10)
    updated = store.import_document("test.md")
    assert updated["document_id"] == first["document_id"]
    assert updated["version_id"] != first["version_id"]
    assert not store.search("证据定位")["hits"]
    with pytest.raises(KBError) as err:
        store.evidence(old_hit["chunk_id"])
    assert err.value.status == 410
    assert store.search("望远镜")["hits"]
    write_source(body=original)
    assert store.import_document("test.md")["version_id"] == first["version_id"]
    assert store.evidence(old_hit["chunk_id"])
    store.delete_document(first["document_id"])
    assert store.stats()["documents"] == 0
    assert not store.search("证据定位")["hits"]
    with pytest.raises(KBError) as err:
        store.evidence(old_hit["chunk_id"])
    assert err.value.status == 410


def test_duplicate_update_keeps_previous_version(store, write_source):
    a = write_source()
    b = "# Other\n## Page 1\n" + "另一个独立文档索引数据内容。" * 10
    write_source("other.md", b)
    first = store.import_document("test.md")
    store.import_document("other.md")
    write_source("test.md", b)
    with pytest.raises(KBError) as err:
        store.import_document("test.md")
    assert err.value.code == "duplicate_update"
    assert (
        next(d for d in store.documents() if d["document_id"] == first["document_id"])["version_id"]
        == first["version_id"]
    )
    write_source("test.md", a)


def test_transaction_rolls_back_index_failure(store, write_source):
    write_source()
    first = store.import_document("test.md")
    hit = store.search("证据定位")["hits"][0]
    with store.connect() as db:
        db.execute(
            "CREATE TRIGGER fail_index BEFORE INSERT ON chunks BEGIN SELECT RAISE(ABORT, 'simulated failure'); END"
        )
    write_source(body="# New\n## Page 1\n" + "transaction failure " * 30)
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        store.import_document("test.md")
    assert store.documents()[0]["version_id"] == first["version_id"]
    assert store.evidence(hit["chunk_id"])
    assert store.search("证据定位")["hits"]


def test_concurrent_identical_imports_have_one_active_version(store, write_source):
    write_source()
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: store.import_document("test.md"), range(3)))
    assert sorted(r["status"] for r in results) == ["imported", "unchanged", "unchanged"]
    assert store.stats()["documents"] == 1


def test_safe_fts_syntax_and_filters(store, write_source):
    write_source()
    first = store.import_document("test.md")
    assert store.search('证据定位 " OR * "; DROP TABLE documents --')["hits"]
    assert store.search("证据定位", document_ids=[first["document_id"]])["hits"]
    assert not store.search("证据定位", document_ids=["doc_unknown"])["hits"]
    assert not store.search("证据定位", document_ids=[])["hits"]
    assert not store.search("***")["hits"]


def test_aliases_are_explicit_and_not_semantic_search(store, write_source):
    write_source()
    store.import_document("test.md")
    assert not store.search("莫诺马赫")["hits"]
    assert store.search("莫诺马赫", expand_aliases=True)["hits"]
    assert tokens("СЕ\u0308МЬ") == tokens("семь")
    assert tokens("中文检索") == ["中文", "文检", "检索"]
    assert "MATCH" not in query_expression('" MATCH *', False)[0]


def test_evaluation_rejects_bad_gold_and_counts_multi_evidence(store, write_source):
    write_source()
    first = store.import_document("test.md")
    gold = {"document_id": first["document_id"], "page": 1, "quote": "证据定位"}
    cases = [{"id": "one", "query": "证据定位", "kind": "single", "evidence": [gold]}]
    report = evaluate(store, cases, False)
    assert report["metrics"]["success_at_5"] == 1
    cases[0]["evidence"][0]["quote"] = "not in source"
    with pytest.raises(ValueError, match="Gold evidence"):
        evaluate(store, cases, False)
