from __future__ import annotations

import time

from .errors import KBError
from .store import Store
from .terminology import GLOSSARY_SHA256, rewrite_query


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    if k < 1:
        raise ValueError("RRF constant must be positive")
    scores = {}
    for ranking in rankings:
        seen = set()
        for rank, chunk_id in enumerate(ranking, 1):
            if chunk_id not in seen:
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
                seen.add(chunk_id)
    return scores


class RetrievalService:
    candidate_depth = 20
    rrf_k = 60

    def __init__(self, store: Store, dense=None):
        self.store, self.dense = store, dense

    @classmethod
    def configured(cls, store: Store):
        if store.settings.embedding_model_dir is None:
            return cls(store)
        from .embedding import E5Encoder
        from .vector import DenseIndex

        return cls(
            store,
            DenseIndex(
                store,
                E5Encoder(store.settings.embedding_model_dir, store.settings.embedding_threads),
            ),
        )

    def close(self):
        if self.dense is not None:
            self.dense.close()

    def state(self):
        return (
            self.dense.state() if self.dense is not None else {"status": "disabled", "ready": False}
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        expand_aliases: bool = False,
        document_ids: list[str] | None = None,
        mode: str = "bm25",
    ) -> dict:
        if mode == "bm25":
            return self.store.search(query, top_k, expand_aliases, document_ids)
        if mode not in {"dense", "hybrid", "hybrid_terms"}:
            raise KBError("invalid_mode", "mode must be bm25, dense, hybrid or hybrid_terms")
        if not query.strip() or len(query) > 1000 or not 1 <= top_k <= 20:
            raise KBError("invalid_query", "Invalid query or top_k")
        if document_ids is not None and len(document_ids) > 20:
            raise KBError("invalid_filter", "At most 20 document IDs are allowed")
        if self.dense is None:
            raise KBError(
                "dense_not_configured",
                "Configure a local embedding model and build vectors first",
                503,
            )
        started = time.perf_counter()
        retrieval_query, matched_terms = (
            rewrite_query(query) if mode == "hybrid_terms" else (query, [])
        )
        is_hybrid = mode in {"hybrid", "hybrid_terms"}
        candidates, metadata = self.dense.search(
            retrieval_query, self.candidate_depth, document_ids
        )
        lexical = (
            self.store.search(retrieval_query, self.candidate_depth, expand_aliases, document_ids)
            if is_hybrid
            else {"hits": [], "alias_expansion": []}
        )
        dense_ids = [h["chunk_id"] for h in candidates]
        lexical_ids = [h["chunk_id"] for h in lexical["hits"]]
        fused = reciprocal_rank_fusion([dense_ids, lexical_ids], self.rrf_k)
        ordered = sorted(fused, key=lambda key: (-fused[key], key)) if is_hybrid else dense_ids
        active = self.store.active_hits(ordered)
        dense_scores = {h["chunk_id"]: h["dense_score"] for h in candidates}
        lexical_scores = {h["chunk_id"]: h["bm25_distance"] for h in lexical["hits"]}
        if self.store.corpus_fingerprint() != metadata["corpus_sha256"]:
            raise KBError(
                "corpus_changed",
                "Corpus changed during search; retry after the current index is built",
                409,
            )
        hits = []
        for chunk_id in ordered[:top_k]:
            if chunk_id not in active:
                raise KBError(
                    "dense_index_inconsistent",
                    "Vector candidate is not active in the source store",
                    503,
                )
            hit = active[chunk_id]
            hit.update(
                dense_score=dense_scores.get(chunk_id),
                bm25_distance=lexical_scores.get(chunk_id),
                rrf_score=fused[chunk_id] if is_hybrid else None,
            )
            hits.append(hit)
        return {
            "retriever": "qdrant_dense"
            if mode == "dense"
            else "bm25_qdrant_rrf" + ("_terms" if mode == "hybrid_terms" else ""),
            "tokenizer_version": "e5-local-onnx-512" if mode == "dense" else "e5-and-bm25",
            "alias_expansion": lexical["alias_expansion"],
            "hits": hits,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "metadata": {
                **metadata,
                "candidate_depth": self.candidate_depth,
                "rrf_k": self.rrf_k if is_hybrid else None,
                "query_terms": matched_terms,
                "glossary_sha256": GLOSSARY_SHA256 if mode == "hybrid_terms" else None,
            },
            "note": "Retrieval candidates only; cosine/RRF scores are not calibrated answer confidence",
        }
