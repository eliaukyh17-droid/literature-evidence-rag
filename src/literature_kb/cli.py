from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .config import Settings
from .evaluation import evaluate
from .retrieval import RetrievalService
from .store import Store


def main():
    parser = argparse.ArgumentParser(description="Local evidence retrieval; no external API calls")
    parser.add_argument("--source-root", type=Path, default=Settings.from_env().source_root)
    parser.add_argument("--data-dir", type=Path, default=Settings.from_env().data_dir)
    parser.add_argument(
        "--embedding-model", type=Path, default=Settings.from_env().embedding_model_dir
    )
    parser.add_argument(
        "--chunk-max-tokens", type=int, default=Settings.from_env().chunk_max_tokens
    )
    parser.add_argument("--local-llm-url", default=Settings.from_env().local_llm_url)
    parser.add_argument("--local-llm-model", default=Settings.from_env().local_llm_model)
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("paths", nargs="+")
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--aliases", action="store_true")
    search.add_argument(
        "--mode", choices=["bm25", "dense", "hybrid", "hybrid_terms"], default="bm25"
    )
    index = sub.add_parser("index-vectors")
    index.add_argument("--batch-size", type=int, default=4)
    evidence = sub.add_parser("evidence")
    evidence.add_argument("chunk_id")
    ask = sub.add_parser("ask")
    ask.add_argument("query")
    ask.add_argument(
        "--mode", choices=["bm25", "dense", "hybrid", "hybrid_terms"], default="hybrid_terms"
    )
    ask.add_argument("--max-new-tokens", type=int, default=384)
    sub.add_parser("stats")
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--cases", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    store = Store(
        Settings(
            args.source_root,
            args.data_dir,
            embedding_model_dir=args.embedding_model,
            chunk_max_tokens=args.chunk_max_tokens,
            local_llm_url=args.local_llm_url,
            local_llm_model=args.local_llm_model,
        )
    )
    if args.command == "ingest":
        result = [store.import_document(path) for path in args.paths]
    elif args.command == "search":
        service = RetrievalService.configured(store)
        try:
            result = service.search(args.query, args.top_k, args.aliases, mode=args.mode)
        finally:
            service.close()
    elif args.command == "index-vectors":
        service = RetrievalService.configured(store)
        try:
            if service.dense is None:
                parser.error("index-vectors requires --embedding-model or KB_EMBEDDING_MODEL_DIR")
            result = service.dense.build(
                args.batch_size, progress=lambda r: print(json.dumps(r), flush=True)
            )
        finally:
            service.close()
    elif args.command == "evidence":
        result = store.evidence(args.chunk_id)
    elif args.command == "ask":
        from .answering import AnswerService
        from .api import configured_generation

        service = RetrievalService.configured(store)

        async def run():
            backend = configured_generation(store.settings)
            try:
                response = await AnswerService(store, service, backend).answer(
                    args.query, mode=args.mode, max_new_tokens=args.max_new_tokens
                )
                return response.model_dump()
            finally:
                if backend is not None:
                    await backend.aclose()

        try:
            result = asyncio.run(run())
        finally:
            service.close()
    elif args.command == "evaluate":
        if args.output.exists():
            raise FileExistsError("Evaluation output exists; choose a new snapshot")
        cases = [
            json.loads(line)
            for line in args.cases.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        result = {"runs": [evaluate(store, cases, False), evaluate(store, cases, True)]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    else:
        result = store.stats()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
