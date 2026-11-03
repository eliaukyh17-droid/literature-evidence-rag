# ruff: noqa: E402 -- skip optional dependencies before importing vector modules.
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("qdrant_client")

from fastapi.testclient import TestClient

from literature_kb.api import create_app
from literature_kb.embedding import mean_pool, prepare_text
from literature_kb.errors import KBError
from literature_kb.retrieval import RetrievalService, reciprocal_rank_fusion
from literature_kb.vector import DenseIndex


class FixtureEncoder:
    """Three orthogonal fixture vectors; ONLY lifecycle tests, never quality evaluation."""

    dimension = 3
    fingerprint = "unit-test-fixture-only-v1"
    descriptor = {"model": "unit-test-fixture-not-a-language-model"}

    def __init__(self):
        self.calls = 0
        self.fail_at = None
        self.truncated_texts = 0

    def encode(self, texts, *, query=False):
        self.calls += 1
        if self.fail_at is not None and self.calls >= self.fail_at:
            raise RuntimeError("simulated encoder interruption")
        return np.asarray(
            [[1.0, 0.0, 0.0] if "cobalt" in t.casefold() else [0.0, 1.0, 0.0] for t in texts],
            dtype=np.float32,
        )


def test_pool_masks_padding_normalizes_and_rejects_zero():
    hidden = np.array([[[3.0, 4.0], [999.0, 999.0]]], dtype=np.float32)
    result = mean_pool(hidden, np.array([[1, 0]]))
    np.testing.assert_allclose(result, [[0.6, 0.8]], atol=1e-7)
    with pytest.raises(ValueError, match="zero"):
        mean_pool(np.zeros((1, 2, 3)), np.array([[1, 1]]))
    assert prepare_text("И\u0306\u00ad\n текст  ") == "Й текст"


def test_rrf_uses_ranks_not_raw_scores_and_does_not_double_count():
    scores = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert scores["b"] > scores["a"] > scores["c"]
    assert scores["a"] == pytest.approx(1 / 61)
    assert reciprocal_rank_fusion([["a", "a"]])["a"] == pytest.approx(1 / 61)


def test_build_resume_reopen_filter_and_hybrid(store, write_source):
    write_source(body="# Cobalt\n## Page 1\n" + "Cobalt source evidence and fixture content. " * 20)
    first = store.import_document("test.md")
    write_source(
        "other.md", "# Amber\n## Page 1\n" + "Amber independent source and fixture content. " * 20
    )
    second = store.import_document("other.md")
    encoder = FixtureEncoder()
    index = DenseIndex(store, encoder)
    try:
        assert index.state()["status"] == "missing"
        manifest = index.build(2)
        assert manifest["chunks"] == store.stats()["chunks"]
        assert index.state()["ready"]
        calls = encoder.calls
        assert index.build(2)["encoded_now"] == 0
        assert encoder.calls == calls
        service = RetrievalService(store, index)
        hit = service.search("cobalt", mode="dense")["hits"][0]
        assert hit["document_id"] == first["document_id"]
        assert hit["dense_score"] == pytest.approx(1)
        filtered = service.search("cobalt", mode="hybrid", document_ids=[second["document_id"]])
        assert filtered["hits"]
        assert all(h["document_id"] == second["document_id"] for h in filtered["hits"])
        assert not service.search("cobalt", mode="dense", document_ids=[])["hits"]
        assert not service.search("cobalt", mode="dense", document_ids=["doc_unknown"])["hits"]
    finally:
        index.close()
    reopened = DenseIndex(store, FixtureEncoder())
    try:
        assert reopened.state()["ready"]
        assert reopened.search("cobalt", 5)[0]
    finally:
        reopened.close()


def test_interruption_does_not_activate_partial_index_and_resume_skips_points(store, write_source):
    write_source()
    store.import_document("test.md")
    encoder = FixtureEncoder()
    encoder.fail_at = 2
    index = DenseIndex(store, encoder)
    try:
        with pytest.raises(RuntimeError, match="interruption"):
            index.build(2)
        assert index.state()["ready"] is False
        encoder.fail_at = None
        manifest = index.build(2)
        assert manifest["resumed_points"] == 2
        assert manifest["encoded_now"] == manifest["chunks"] - 2
        assert index.state()["ready"]
    finally:
        index.close()


def test_changed_deleted_corpus_and_encoder_reject_stale_index(store, write_source):
    write_source()
    doc = store.import_document("test.md")
    encoder = FixtureEncoder()
    index = DenseIndex(store, encoder)
    try:
        index.build()
        with store.connect() as db:
            old_pointer = db.execute(
                "SELECT value FROM metadata WHERE key='dense_active'"
            ).fetchone()[0]
        write_source(body="# New version\n## Page 1\n" + "Cobalt updated evidence content. " * 20)
        store.import_document("test.md")
        with pytest.raises(KBError) as err:
            index.search("cobalt", 5)
        assert err.value.status == 503
        encoder.fail_at = encoder.calls + 1
        with pytest.raises(RuntimeError):
            index.build()
        with store.connect() as db:
            assert (
                db.execute("SELECT value FROM metadata WHERE key='dense_active'").fetchone()[0]
                == old_pointer
            )
        encoder.fail_at = None
        index.build()
        assert index.state()["ready"]
        encoder.fingerprint = "unit-test-different-model"
        assert index.state()["status"] == "stale"
        encoder.fingerprint = "unit-test-fixture-only-v1"
        store.delete_document(doc["document_id"])
        with pytest.raises(KBError):
            index.search("cobalt", 5)
    finally:
        index.close()


def test_mutation_during_build_is_not_activated(store, write_source):
    write_source()
    store.import_document("test.md")
    index = DenseIndex(store, FixtureEncoder())
    changed = False

    def mutate(_):
        nonlocal changed
        if not changed:
            changed = True
            write_source("new.md", "# Different\n## Page 1\n" + "New fixture corpus content. " * 20)
            store.import_document("new.md")

    try:
        with pytest.raises(KBError) as err:
            index.build(2, mutate)
        assert err.value.code == "corpus_changed"
        assert index.state()["ready"] is False
    finally:
        index.close()


def test_payload_corruption_is_not_silently_accepted(store, write_source):
    from qdrant_client import models

    write_source()
    store.import_document("test.md")
    index = DenseIndex(store, FixtureEncoder())
    try:
        manifest = index.build()
        records, _ = index.client.scroll(manifest["collection"], limit=1)
        index.client.set_payload(
            manifest["collection"],
            payload={"text_sha256": "wrong"},
            points=models.PointIdsList(points=[records[0].id]),
        )
        with pytest.raises(ValueError, match="identity"):
            index.build()
    finally:
        index.close()


def test_api_modes_validation_and_stale_failure(settings, write_source):
    write_source()

    def factory(store):
        return RetrievalService(store, DenseIndex(store, FixtureEncoder()))

    app = create_app(settings, retrieval_factory=factory)
    with TestClient(app) as client:
        doc = client.post("/v1/documents/import", json={"relative_path": "test.md"}).json()
        assert client.post("/v1/search", json={"query": "x", "mode": "dense"}).status_code == 503
        app.state.retrieval.dense.build()
        result = client.post("/v1/search", json={"query": "x", "mode": "hybrid"})
        assert result.status_code == 200
        assert result.json()["hits"][0]["rrf_score"] is not None
        assert result.json()["metadata"]["rrf_k"] == 60
        terms_result = client.post("/v1/search", json={"query": "莫诺马赫", "mode": "hybrid_terms"})
        assert terms_result.status_code == 200
        assert terms_result.json()["metadata"]["query_terms"][0]["source"] == "莫诺马赫"
        assert client.get("/v1/health").json()["dense_index"]["ready"]
        assert client.post("/v1/search", json={"query": "x", "mode": "invented"}).status_code == 422
        client.delete("/v1/documents/" + doc["document_id"])
        assert client.post("/v1/search", json={"query": "x", "mode": "dense"}).status_code == 503


def test_vector_disabled_is_explicit_not_lexical_fallback(settings):
    with TestClient(create_app(settings)) as client:
        result = client.post("/v1/search", json={"query": "question", "mode": "dense"})
        assert result.status_code == 503
        assert result.json()["error"]["code"] == "dense_not_configured"
