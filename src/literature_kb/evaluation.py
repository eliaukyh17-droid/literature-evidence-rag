from __future__ import annotations

from collections import Counter

from .store import Store


def evaluate(
    store: Store, cases: list[dict], expand_aliases: bool, *, searcher=None, mode: str = "bm25"
) -> dict:
    """Development retrieval evaluation. Gold quotes must exist verbatim in active pages.

    Evidence is judged by (document, page, exact quote in retrieved chunk). It is not
    an answer correctness or multilingual semantic quality benchmark.
    """
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Cases must be nonempty with unique IDs")
    rows, reciprocal, success5, success10 = [], [], [], []
    recall5, recall10 = [], []
    with store.connect() as db:
        for case in cases:
            gold = case.get("evidence", [])
            if not gold and case.get("kind") != "no_answer":
                raise ValueError("Answerable cases require verified evidence")
            for item in gold:
                page = db.execute(
                    "SELECT p.text FROM pages p JOIN documents d ON d.active_version=p.version_id "
                    "WHERE d.document_id=? AND d.deleted=0 AND p.page_number=?",
                    (item["document_id"], item["page"]),
                ).fetchone()
                if not item["quote"].strip() or page is None or item["quote"] not in page[0]:
                    raise ValueError(
                        f"Gold evidence does not match the active snapshot: {case['id']}"
                    )
            if searcher is None:
                result = store.search(case["query"], top_k=10, expand_aliases=expand_aliases)
            else:
                result = searcher.search(
                    case["query"], top_k=10, expand_aliases=expand_aliases, mode=mode
                )
            hits = result["hits"]
            ranks = []
            for item in gold:
                rank = next(
                    (
                        i
                        for i, hit in enumerate(hits, 1)
                        if hit["document_id"] == item["document_id"]
                        and hit["page_start"] == item["page"]
                        and item["quote"] in hit["text"]
                    ),
                    None,
                )
                ranks.append(rank)
            if gold:
                first = min((rank for rank in ranks if rank is not None), default=None)
                reciprocal.append(1 / first if first else 0)
                success5.append(bool(first and first <= 5))
                success10.append(bool(first and first <= 10))
                recall5.append(sum(bool(r and r <= 5) for r in ranks) / len(gold))
                recall10.append(sum(bool(r and r <= 10) for r in ranks) / len(gold))
            rows.append(
                {
                    "id": case["id"],
                    "kind": case["kind"],
                    "evidence_ranks": ranks,
                    "hits": [
                        {
                            "chunk_id": h["chunk_id"],
                            "document_id": h["document_id"],
                            "page": h["page_start"],
                        }
                        for h in hits
                    ],
                    "elapsed_ms": result["elapsed_ms"],
                    "alias_expansion": result["alias_expansion"],
                    "retrieval_metadata": result.get("metadata"),
                }
            )
    if not reciprocal:
        raise ValueError("At least one answerable case is required")

    def mean(values):
        return round(sum(values) / len(values), 6)

    return {
        "schema": "retrieval-development-v1",
        "mode": mode + ("_aliases" if expand_aliases else ""),
        "case_count": len(cases),
        "answerable_count": len(reciprocal),
        "kind_counts": dict(Counter(c["kind"] for c in cases)),
        "metrics": {
            "success_at_5": mean(success5),
            "success_at_10": mean(success10),
            "evidence_recall_at_5": mean(recall5),
            "evidence_recall_at_10": mean(recall10),
            "mrr_at_10": mean(reciprocal),
        },
        "corpus": store.stats(),
        "cases": rows,
        "limitations": [
            "Development queries only; not an independent blind test or answer accuracy score",
            "No-answer cases are observations only; lexical hits do not establish support",
            "Known page markers/gold evidence do not independently verify PDF/OCR accuracy",
            "Explicit aliases are not embeddings or general Chinese-to-Russian translation",
        ],
    }
