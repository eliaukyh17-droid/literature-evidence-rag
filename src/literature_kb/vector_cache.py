from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from .embedding import prepare_text
from .parsing import digest


def passage_key(row: dict) -> str:
    return digest("passage: " + prepare_text(row["title"] + "\n" + row["text"]))


class VectorCache:
    """Private derived-data cache, isolated by the complete encoder fingerprint."""

    def __init__(self, path: Path, embedding_id: str, dimension: int):
        self.path, self.embedding_id, self.dimension = path, embedding_id, dimension
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS embeddings (encoder TEXT, passage TEXT, dimension INTEGER, vector BLOB, PRIMARY KEY(encoder,passage))"
            )

    def _validate(self, vector):
        array = np.asarray(vector, dtype="<f4")
        if (
            array.shape != (self.dimension,)
            or not np.isfinite(array).all()
            or not np.isclose(np.linalg.norm(array), 1, atol=1e-4)
        ):
            raise ValueError("Cached embedding failed dimension/finite/norm validation")
        return array

    def get(self, key: str):
        with sqlite3.connect(self.path, timeout=15) as db:
            row = db.execute(
                "SELECT dimension,vector FROM embeddings WHERE encoder=? AND passage=?",
                (self.embedding_id, key),
            ).fetchone()
        if row is None:
            return None
        if row[0] != self.dimension:
            raise ValueError("Embedding cache dimension mismatch")
        return self._validate(np.frombuffer(row[1], dtype="<f4"))

    def put(self, values: list[tuple[str, np.ndarray]]):
        records = [
            (self.embedding_id, key, self.dimension, self._validate(vec).tobytes())
            for key, vec in values
        ]
        with sqlite3.connect(self.path, timeout=15) as db:
            db.executemany("INSERT OR IGNORE INTO embeddings VALUES(?,?,?,?)", records)
