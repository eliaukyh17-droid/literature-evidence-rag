from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("qdrant_client")


def test_cache_encoder_isolation_and_corruption_detection(tmp_path):
    import sqlite3

    from literature_kb.vector_cache import VectorCache

    path = tmp_path / "cache.sqlite3"
    a = VectorCache(path, "encoder-a", 3)
    b = VectorCache(path, "encoder-b", 3)
    a.put([("same-passage", np.array([1, 0, 0]))])
    assert b.get("same-passage") is None
    np.testing.assert_allclose(a.get("same-passage"), [1, 0, 0])
    with sqlite3.connect(path) as db:
        db.execute(
            "UPDATE embeddings SET vector=? WHERE encoder='encoder-a'",
            (np.array([np.nan, 0, 0], dtype="<f4").tobytes(),),
        )
    with pytest.raises(ValueError, match="validation"):
        a.get("same-passage")


def test_recipe_revision_can_reuse_identical_encoder_input(store, settings, write_source):
    from test_vector import FixtureEncoder

    from literature_kb.store import Store
    from literature_kb.vector import DenseIndex

    write_source(body="# Cobalt\n## Page 1\n" + "Cobalt source fact. " * 4)
    old = store.import_document("test.md")
    first = DenseIndex(store, FixtureEncoder())
    try:
        first.build()
    finally:
        first.close()
    revised = Store(replace(settings, chunk_chars=220))
    new = revised.import_document("test.md")
    assert old["version_id"] != new["version_id"]
    encoder = FixtureEncoder()
    second = DenseIndex(revised, encoder)
    try:
        manifest = second.build()
        assert manifest["cache_hits"] == 1 and manifest["encoded_now"] == 0
        assert encoder.calls == 0
    finally:
        second.close()
