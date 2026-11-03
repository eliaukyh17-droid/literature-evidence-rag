from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import anyio
from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .answering import AnswerResponse, AnswerService
from .config import Settings
from .errors import KBError
from .retrieval import RetrievalService
from .store import Store

logger = logging.getLogger("literature_kb.requests")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImportRequest(StrictModel):
    relative_path: str = Field(min_length=1, max_length=500)


class SearchRequest(StrictModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    expand_aliases: bool = False
    mode: Literal["bm25", "dense", "hybrid", "hybrid_terms"] = "bm25"
    document_ids: list[str] | None = Field(default=None, max_length=20)

    @field_validator("query")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("query cannot be blank")
        return value


class Hit(BaseModel):
    chunk_id: str
    document_id: str
    version_id: str
    title: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    text: str
    text_sha256: str
    flags: list[str]
    bm25_distance: float | None = None
    dense_score: float | None = None
    rrf_score: float | None = None
    evidence_url: str


class SearchResponse(BaseModel):
    retriever: str
    tokenizer_version: str
    alias_expansion: list[str]
    hits: list[Hit]
    elapsed_ms: float
    note: str
    metadata: dict | None = None


class AnswerRequest(StrictModel):
    query: str = Field(min_length=1, max_length=1000)
    mode: Literal["bm25", "dense", "hybrid", "hybrid_terms"] = "hybrid_terms"
    top_k: int = Field(default=5, ge=1, le=10)
    max_new_tokens: int = Field(default=384, ge=128, le=768)
    document_ids: list[str] | None = Field(default=None, max_length=20)

    @field_validator("query")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("query cannot be blank")
        return value


def configured_generation(settings):
    if settings.local_llm_url is None:
        return None
    from .generation import LocalLlamaBackend

    return LocalLlamaBackend(
        settings.local_llm_url, settings.local_llm_model, settings.generation_context_tokens
    )


def create_app(
    settings: Settings | None = None,
    *,
    retrieval_factory=RetrievalService.configured,
    generation_factory=configured_generation,
) -> FastAPI:
    config = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        app.state.store = Store(config)
        service = retrieval_factory(app.state.store)
        app.state.retrieval = service
        backend = generation_factory(config)
        app.state.answering = AnswerService(
            app.state.store, service, backend, config.answer_timeout_seconds
        )
        try:
            yield
        finally:
            if backend is not None:
                await backend.aclose()
            service.close()

    app = FastAPI(
        title="Literature Evidence API",
        version="0.4.0",
        lifespan=lifespan,
        description="Single-user literature retrieval and grounded answers with a local model. Bind to 127.0.0.1.",
    )

    @app.middleware("http")
    async def request_observation(request: Request, call_next):
        candidate = request.headers.get("X-Request-ID", "")
        request_id = (
            candidate if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", candidate) else uuid.uuid4().hex
        )
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        # No prompts, document text, keys, absolute paths or request bodies in logs.
        route = request.scope.get("route")
        logger.info(
            "request_id=%s method=%s route=%s status=%s elapsed_ms=%.3f",
            request_id,
            request.method,
            getattr(route, "path", "unmatched"),
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response

    @app.exception_handler(KBError)
    async def kb_error(request: Request, exc: KBError):
        logger.info("request_id=%s error_code=%s", request.state.request_id, exc.code)
        return JSONResponse(
            status_code=exc.status,
            content={
                "error": {"code": exc.code, "message": exc.message},
                "request_id": request.state.request_id,
            },
        )

    router = APIRouter(prefix="/v1")

    @router.get("/health")
    def health(request: Request):
        stats = request.app.state.store.stats()
        stats["answer_generation_enabled"] = request.app.state.answering.backend is not None
        stats["demo_mode"] = getattr(request.app.state.answering.backend, "demo_mode", False)
        stats["answer_in_progress"] = request.app.state.answering.lock.locked()
        state = request.app.state.retrieval.state()
        if state["ready"]:
            stats["retrievers"] += ["qdrant_dense", "bm25_qdrant_rrf", "bm25_qdrant_rrf_terms"]
        return {"status": "ok", **stats, "dense_index": state}

    @router.get("/documents")
    def documents(request: Request):
        return {"documents": request.app.state.store.documents()}

    @router.post("/documents/import")
    def import_document(body: ImportRequest, request: Request):
        # Synchronous pilot import. Durable background ingestion is a later milestone.
        return request.app.state.store.import_document(body.relative_path)

    @router.delete("/documents/{document_id}", status_code=204)
    def delete_document(document_id: str, request: Request):
        request.app.state.store.delete_document(document_id)
        return Response(status_code=204)

    @router.post("/search", response_model=SearchResponse)
    def search(body: SearchRequest, request: Request):
        return request.app.state.retrieval.search(**body.model_dump())

    @router.get("/evidence/{chunk_id}")
    def evidence(chunk_id: str, request: Request):
        return request.app.state.store.evidence(chunk_id)

    @router.post("/answer", response_model=AnswerResponse)
    async def answer(body: AnswerRequest, request: Request):
        task = asyncio.create_task(
            request.app.state.answering.answer(
                **body.model_dump(), request_id=request.state.request_id
            )
        )

        async def disconnected():
            while not await request.is_disconnected():
                await asyncio.sleep(0.1)

        monitor = asyncio.create_task(disconnected())
        try:
            done, _ = await asyncio.wait({task, monitor}, return_when=asyncio.FIRST_COMPLETED)
            if monitor in done and task not in done:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise KBError(
                    "client_disconnected", "Client disconnected; generation request cancelled", 499
                )
            return await task
        finally:
            for running in (task, monitor):
                if not running.done():
                    running.cancel()
            await asyncio.gather(task, monitor, return_exceptions=True)

    @router.post(
        "/answer/stream",
        response_class=StreamingResponse,
        responses={200: {"content": {"text/event-stream": {}}}},
    )
    async def answer_stream(body: AnswerRequest, request: Request):
        if request.headers.get("last-event-id"):
            raise KBError("stream_resume_unavailable", "Start a new question to open a stream", 400)
        request_id = request.state.request_id

        async def events():
            queue = asyncio.Queue(maxsize=16)
            started = time.perf_counter()

            async def emit(event, data):
                await queue.put((event, data))

            async def produce():
                try:
                    result = await request.app.state.answering.answer(
                        **body.model_dump(), request_id=request_id, on_event=emit
                    )
                    await emit("done", result.model_dump())
                except KBError as exc:
                    logger.info("request_id=%s stream_error=%s", request_id, exc.code)
                    await emit(
                        "error", {"code": exc.code, "message": exc.message, "status": exc.status}
                    )
                except asyncio.CancelledError:
                    logger.info("request_id=%s stream_cancelled=true", request_id)
                    raise
                except Exception:
                    logger.error("request_id=%s stream_error=internal_error", request_id)
                    await emit(
                        "error",
                        {
                            "code": "internal_error",
                            "message": "Answer stream failed",
                            "status": 500,
                        },
                    )

            task = asyncio.create_task(produce())
            sequence = 0
            try:
                event, data = "started", {}
                while True:
                    sequence += 1
                    envelope = {
                        "request_id": request_id,
                        "sequence": sequence,
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                        **data,
                    }
                    yield (
                        f"id: {request_id}:{sequence}\nevent: {event}\n"
                        + "data: "
                        + json.dumps(envelope, ensure_ascii=False)
                        + "\n\n"
                    )
                    if event in {"done", "error"}:
                        break
                    try:
                        event, data = await asyncio.wait_for(queue.get(), timeout=15)
                    except TimeoutError:
                        event, data = "ping", {}
            finally:
                if not task.done():
                    task.cancel()
                with anyio.CancelScope(shield=True):
                    await asyncio.gather(task, return_exceptions=True)
                logger.info(
                    "request_id=%s stream_closed=true elapsed_ms=%.3f",
                    request_id,
                    (time.perf_counter() - started) * 1000,
                )

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
                "X-Content-Type-Options": "nosniff",
            },
        )

    app.include_router(router)
    web = Path(__file__).with_name("web")
    app.mount("/assets", StaticFiles(directory=web), name="assets")

    @app.get("/", include_in_schema=False)
    def workbench():
        return FileResponse(
            web / "index.html",
            headers={
                "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
                "Cache-Control": "no-store",
            },
        )

    return app
