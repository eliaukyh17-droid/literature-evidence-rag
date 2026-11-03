from __future__ import annotations

import json
import threading
import time
import uuid

import numpy as np
from qdrant_client import QdrantClient, models

from .errors import KBError
from .parsing import digest
from .store import Store
from .vector_cache import VectorCache, passage_key


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "literature-kb:" + chunk_id))


class DenseIndex:
    """Immutable corpus generations, with a SQLite activation pointer after verification.

    Qdrant local mode owns an exclusive on-disk lock. Use ONE process/worker;
    stop the API before running the index CLI. No remote/cloud inference is enabled.
    """

    def __init__(self, store: Store, encoder):
        self.store, self.encoder = store, encoder
        self.lock = threading.RLock()
        self.cache = VectorCache(
            store.settings.data_dir / "embedding_cache.sqlite3",
            encoder.fingerprint,
            encoder.dimension,
        )
        self.client = QdrantClient(
            path=str(store.settings.data_dir / "qdrant"),
            force_disable_check_same_thread=True,
            cloud_inference=False,
        )

    def close(self):
        with self.lock:
            self.client.close()

    def state(self) -> dict:
        with self.store.connect() as db:
            row = db.execute("SELECT value FROM metadata WHERE key='dense_active'").fetchone()
        if row is None:
            return {"status": "missing", "ready": False}
        state = json.loads(row[0])
        ready = (
            state["embedding_id"] == self.encoder.fingerprint
            and state["corpus_sha256"] == self.store.corpus_fingerprint()
        )
        ready = ready and self.client.collection_exists(state["collection"])
        if ready:
            ready = self.client.count(state["collection"], exact=True).count == state["chunks"]
        return {**state, "status": "ready" if ready else "stale", "ready": ready}

    def build(self, batch_size: int = 4, progress=None) -> dict:
        if not 1 <= batch_size <= 16:
            raise ValueError("Index batch_size must be 1-16")
        with self.lock:
            rows = self.store.active_chunks()
            if not rows:
                raise KBError("empty_corpus", "Import documents before building vectors", 409)
            corpus_sha = self.store.fingerprint_chunks(rows)
            collection = "evidence_" + digest(corpus_sha + self.encoder.fingerprint)[:32]
            expected = {point_id(r["chunk_id"]): r for r in rows}
            if not self.client.collection_exists(collection):
                self.client.create_collection(
                    collection,
                    vectors_config=models.VectorParams(
                        size=self.encoder.dimension, distance=models.Distance.COSINE
                    ),
                )
            present = {}
            offset = None
            while True:
                points, offset = self.client.scroll(
                    collection, limit=128, offset=offset, with_payload=True
                )
                present.update({str(p.id): p.payload for p in points})
                if offset is None:
                    break
            for pid, payload in present.items():
                if pid not in expected or payload != self._payload(expected[pid]):
                    raise ValueError(
                        "Partial collection identity differs from the immutable generation"
                    )
            remaining = [r for pid, r in expected.items() if pid not in present]
            started = time.perf_counter()
            before = self.encoder.truncated_texts
            encoded_count, cache_hits = 0, 0
            for start in range(0, len(remaining), batch_size):
                batch = remaining[start : start + batch_size]
                vectors = np.empty((len(batch), self.encoder.dimension), dtype=np.float32)
                missing = []
                for i, row in enumerate(batch):
                    cached = self.cache.get(passage_key(row))
                    if cached is None:
                        missing.append(i)
                    else:
                        vectors[i] = cached
                        cache_hits += 1
                if missing:
                    fresh = self.encoder.encode(
                        [batch[i]["title"] + "\n" + batch[i]["text"] for i in missing]
                    )
                    if fresh.shape != (len(missing), self.encoder.dimension):
                        raise ValueError("Encoder returned invalid batch shape")
                    for i, vec in zip(missing, fresh, strict=True):
                        vectors[i] = vec
                    self.cache.put([(passage_key(batch[i]), vectors[i]) for i in missing])
                    encoded_count += len(missing)
                if (
                    vectors.shape != (len(batch), self.encoder.dimension)
                    or not np.isfinite(vectors).all()
                ):
                    raise ValueError("Encoder returned invalid vector batch")
                self.client.upsert(
                    collection,
                    points=[
                        models.PointStruct(
                            id=point_id(r["chunk_id"]), vector=v.tolist(), payload=self._payload(r)
                        )
                        for r, v in zip(batch, vectors, strict=True)
                    ],
                    wait=True,
                )
                if progress:
                    progress(
                        {
                            "event": "vector_index_progress",
                            "completed": len(present) + min(start + batch_size, len(remaining)),
                            "total": len(rows),
                            "elapsed_seconds": round(time.perf_counter() - started, 2),
                        }
                    )
            # Verify the entire generation, including restored points, before activation.
            verified, offset = set(), None
            while True:
                points, offset = self.client.scroll(
                    collection, limit=128, offset=offset, with_payload=True, with_vectors=True
                )
                for point in points:
                    pid = str(point.id)
                    vector = np.asarray(point.vector)
                    if (
                        pid not in expected
                        or point.payload != self._payload(expected[pid])
                        or vector.shape != (self.encoder.dimension,)
                        or not np.isfinite(vector).all()
                        or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4)
                    ):
                        raise ValueError("Vector generation failed full verification")
                    verified.add(pid)
                if offset is None:
                    break
            if verified != set(expected):
                raise ValueError("Vector generation is incomplete")
            manifest = {
                "collection": collection,
                "corpus_sha256": corpus_sha,
                "embedding_id": self.encoder.fingerprint,
                "model": self.encoder.descriptor,
                "chunks": len(rows),
                "resumed_points": len(present),
                "encoded_now": encoded_count,
                "cache_hits": cache_hits,
                "truncated_texts_this_run": self.encoder.truncated_texts - before,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "mode": "qdrant_local_exact_cosine",
                "batch_size": batch_size,
                "encoder_threads": getattr(self.encoder, "threads", None),
                "api_calls": 0,
            }
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                if self.store.corpus_fingerprint(db) != corpus_sha:
                    raise KBError(
                        "corpus_changed",
                        "Corpus changed during indexing; generation was not activated",
                        409,
                    )
                db.execute(
                    "INSERT OR REPLACE INTO metadata(key,value) VALUES('dense_active',?)",
                    (json.dumps(manifest, sort_keys=True),),
                )
            return manifest

    def _payload(self, row):
        return {
            "chunk_id": row["chunk_id"],
            "document_id": row["document_id"],
            "version_id": row["version_id"],
            "text_sha256": row["text_sha256"],
            "embedding_id": self.encoder.fingerprint,
        }

    def search(
        self, query: str, limit: int, document_ids: list[str] | None = None
    ) -> tuple[list[dict], dict]:
        with self.lock:
            state = self.state()
            if not state["ready"]:
                raise KBError(
                    "dense_index_not_ready",
                    "Build vectors for the current corpus/model before dense search",
                    503,
                )
            if document_ids == []:
                return [], {"corpus_sha256": state["corpus_sha256"], "query_truncated": False}
            before = self.encoder.truncated_texts
            vector = self.encoder.encode([query], query=True)[0]
            filters = (
                models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchAny(any=document_ids)
                        )
                    ]
                )
                if document_ids
                else None
            )
            points = self.client.query_points(
                state["collection"],
                query=vector.tolist(),
                query_filter=filters,
                limit=limit,
                with_payload=True,
            ).points
            rows = [{"chunk_id": p.payload["chunk_id"], "dense_score": p.score} for p in points]
            return rows, {
                "corpus_sha256": state["corpus_sha256"],
                "embedding_id": state["embedding_id"],
                "query_truncated": self.encoder.truncated_texts > before,
            }
