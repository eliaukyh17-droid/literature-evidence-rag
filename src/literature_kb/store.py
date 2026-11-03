from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict

from .config import Settings
from .errors import KBError
from .lexical import TOKENIZER_VERSION, query_expression, tokens
from .parsing import ChunkRecipe, digest, parse_document, safe_source

SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
 document_id TEXT PRIMARY KEY, relative_path TEXT NOT NULL UNIQUE,
 active_version TEXT, deleted INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS versions (
 version_id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(document_id),
 source_sha256 TEXT NOT NULL, recipe_sha256 TEXT NOT NULL, title TEXT NOT NULL,
 source_name TEXT, page_count INTEGER NOT NULL, chunk_count INTEGER NOT NULL,
 skipped_pages TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS pages (
 version_id TEXT NOT NULL REFERENCES versions(version_id), page_number INTEGER NOT NULL,
 char_start INTEGER NOT NULL, text TEXT NOT NULL, flags TEXT NOT NULL,
 PRIMARY KEY (version_id,page_number));
CREATE TABLE IF NOT EXISTS chunks (
 row_id INTEGER PRIMARY KEY, chunk_id TEXT NOT NULL UNIQUE,
 version_id TEXT NOT NULL REFERENCES versions(version_id), page_number INTEGER NOT NULL,
 page_char_start INTEGER NOT NULL, page_char_end INTEGER NOT NULL,
 char_start INTEGER NOT NULL, char_end INTEGER NOT NULL, text TEXT NOT NULL,
 text_sha256 TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS chunk_version ON chunks(version_id);
CREATE INDEX IF NOT EXISTS version_document ON versions(document_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(title_tokens, body_tokens);
"""


class Store:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.source_root = settings.source_root.resolve(strict=True)
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = settings.data_dir / "kb.sqlite3"
        self.token_counter = None
        if settings.chunk_max_tokens is not None:
            if settings.embedding_model_dir is None:
                raise ValueError("Token-aware chunking requires a local tokenizer directory")
            from .token_budget import E5TokenBudget

            self.token_counter = E5TokenBudget(settings.embedding_model_dir)
        self.recipe = ChunkRecipe(
            settings.chunk_chars,
            settings.overlap_chars,
            max_tokens=settings.chunk_max_tokens,
            tokenizer_sha256=self.token_counter.sha256 if self.token_counter else None,
        )
        self.recipe_sha = digest(self.recipe.fingerprint + TOKENIZER_VERSION)
        with self.connect() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in {0, 1}:
                raise RuntimeError("Unsupported database schema; do not overwrite this database")
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
            previous = db.execute("SELECT value FROM metadata WHERE key='source_root'").fetchone()
            if previous and previous[0] != str(self.source_root):
                raise ValueError(
                    "Database belongs to a different source root; use a separate data directory"
                )
            db.execute(
                "INSERT OR IGNORE INTO metadata VALUES('source_root',?)", (str(self.source_root),)
            )
            db.execute("PRAGMA user_version=1")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db_path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def import_document(self, relative_path: str) -> dict:
        path = safe_source(self.source_root, relative_path, self.settings.max_source_bytes)
        canonical_path = path.relative_to(self.source_root).as_posix()
        raw = path.read_bytes()
        if len(raw) > self.settings.max_source_bytes:
            raise KBError("source_too_large", "Source exceeds the configured byte limit", 413)
        try:
            text = raw.decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError as exc:
            raise KBError("invalid_encoding", "Markdown source must be UTF-8") from exc
        source_sha = digest(raw)
        doc_id = "doc_" + digest(canonical_path)[:24]
        version_id = "ver_" + digest(doc_id + source_sha + self.recipe_sha)[:32]
        # Parsing happens before mutation. A bad update cannot deactivate a good version.
        parsed = parse_document(text, self.recipe, self.token_counter)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            duplicate = db.execute(
                "SELECT d.document_id FROM documents d JOIN versions v ON v.version_id=d.active_version "
                "WHERE d.deleted=0 AND v.source_sha256=? AND d.document_id<>? LIMIT 1",
                (source_sha, doc_id),
            ).fetchone()
            existing = db.execute(
                "SELECT * FROM documents WHERE document_id=?", (doc_id,)
            ).fetchone()
            if duplicate:
                if existing and not existing["deleted"]:
                    raise KBError(
                        "duplicate_update", "Update duplicates another active document", 409
                    )
                return {
                    "status": "duplicate",
                    "document_id": duplicate[0],
                    "source_sha256": source_sha,
                }
            if existing and existing["active_version"] == version_id and not existing["deleted"]:
                return self._import_result(db, doc_id, version_id, "unchanged")
            db.execute(
                "INSERT OR IGNORE INTO documents(document_id,relative_path) VALUES(?,?)",
                (doc_id, canonical_path),
            )
            if not db.execute(
                "SELECT 1 FROM versions WHERE version_id=?", (version_id,)
            ).fetchone():
                db.execute(
                    "INSERT INTO versions(version_id,document_id,source_sha256,recipe_sha256,title,source_name,"
                    "page_count,chunk_count,skipped_pages) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        version_id,
                        doc_id,
                        source_sha,
                        self.recipe_sha,
                        parsed.title,
                        parsed.source_name,
                        len(parsed.pages),
                        len(parsed.chunks),
                        json.dumps(parsed.skipped_pages),
                    ),
                )
                db.executemany(
                    "INSERT INTO pages VALUES(?,?,?,?,?)",
                    [
                        (version_id, p.number, p.char_start, p.text, json.dumps(p.flags))
                        for p in parsed.pages
                    ],
                )
                for c in parsed.chunks:
                    chunk_id = "chk_" + digest(f"{version_id}:{c.char_start}:{c.char_end}")[:32]
                    db.execute(
                        "INSERT INTO chunks(chunk_id,version_id,page_number,page_char_start,page_char_end,"
                        "char_start,char_end,text,text_sha256) VALUES(?,?,?,?,?,?,?,?,?)",
                        (
                            chunk_id,
                            version_id,
                            c.page,
                            c.page_char_start,
                            c.page_char_end,
                            c.char_start,
                            c.char_end,
                            c.text,
                            c.text_sha256,
                        ),
                    )
            self._remove_fts(db, doc_id)
            title_tokens = " ".join(tokens(parsed.title))
            db.executemany(
                "INSERT INTO chunk_fts(rowid,title_tokens,body_tokens) VALUES(?,?,?)",
                [
                    (c["row_id"], title_tokens, " ".join(tokens(c["text"])))
                    for c in db.execute(
                        "SELECT row_id,text FROM chunks WHERE version_id=? ORDER BY row_id",
                        (version_id,),
                    )
                ],
            )
            # One commit activates the new pages, chunks and index together.
            db.execute(
                "UPDATE documents SET active_version=?,deleted=0 WHERE document_id=?",
                (version_id, doc_id),
            )
            return self._import_result(
                db, doc_id, version_id, "updated" if existing else "imported"
            )

    @staticmethod
    def _remove_fts(db, doc_id):
        db.execute(
            "DELETE FROM chunk_fts WHERE rowid IN (SELECT c.row_id FROM chunks c JOIN versions v "
            "ON c.version_id=v.version_id WHERE v.document_id=?)",
            (doc_id,),
        )

    @staticmethod
    def _import_result(db, doc_id, version_id, status):
        row = db.execute("SELECT * FROM versions WHERE version_id=?", (version_id,)).fetchone()
        return {
            "status": status,
            "document_id": doc_id,
            "version_id": version_id,
            "source_sha256": row["source_sha256"],
            "page_count": row["page_count"],
            "chunk_count": row["chunk_count"],
            "skipped_pages": json.loads(row["skipped_pages"]),
        }

    def documents(self) -> list[dict]:
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT d.document_id,v.version_id,v.title,v.source_sha256,v.page_count,v.chunk_count,"
                    "v.created_at FROM documents d JOIN versions v ON v.version_id=d.active_version "
                    "WHERE d.deleted=0 ORDER BY d.document_id"
                )
            ]

    def delete_document(self, doc_id: str) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute(
                "SELECT 1 FROM documents WHERE document_id=? AND deleted=0", (doc_id,)
            ).fetchone():
                raise KBError("document_not_found", "Active document not found", 404)
            self._remove_fts(db, doc_id)
            db.execute("UPDATE documents SET deleted=1 WHERE document_id=?", (doc_id,))
        # Soft deletion for traceability. Old private snapshots stay on disk until a future purge.

    def search(
        self,
        query: str,
        top_k: int = 5,
        expand_aliases: bool = False,
        document_ids: list[str] | None = None,
    ) -> dict:
        if not query.strip() or len(query) > 1000 or not 1 <= top_k <= 20:
            raise KBError(
                "invalid_query", "Query must contain 1-1000 characters and top_k must be 1-20"
            )
        if document_ids is not None and len(document_ids) > 20:
            raise KBError("invalid_filter", "At most 20 document IDs are allowed")
        started = time.perf_counter()
        expression, expansions = query_expression(query, expand_aliases)
        rows = []
        if expression:
            filters, params = "", [expression]
            if document_ids is not None:
                if not document_ids:
                    filters = " AND 0"
                else:
                    filters = " AND d.document_id IN (" + ",".join("?" for _ in document_ids) + ")"
                    params.extend(document_ids)
            with self.connect() as db:
                rows = db.execute(
                    "SELECT c.chunk_id,d.document_id,c.version_id,v.title,c.page_number AS page_start,"
                    "c.page_number AS page_end,c.char_start,c.char_end,c.text,c.text_sha256,p.flags,"
                    "bm25(chunk_fts,0.2,1.0) AS bm25_distance FROM chunk_fts "
                    "JOIN chunks c ON c.row_id=chunk_fts.rowid JOIN versions v ON v.version_id=c.version_id "
                    "JOIN documents d ON d.active_version=c.version_id AND d.document_id=v.document_id "
                    "JOIN pages p ON p.version_id=c.version_id AND p.page_number=c.page_number "
                    "WHERE chunk_fts MATCH ? AND d.deleted=0"
                    + filters
                    + " ORDER BY bm25_distance,c.chunk_id LIMIT ?",
                    (*params, top_k),
                ).fetchall()
        hits = []
        for row in rows:
            hit = dict(row)
            hit["flags"] = json.loads(hit["flags"])
            hit["evidence_url"] = "/v1/evidence/" + hit["chunk_id"]
            hits.append(hit)
        return {
            "retriever": "sqlite_fts5_bm25",
            "tokenizer_version": TOKENIZER_VERSION,
            "alias_expansion": expansions,
            "hits": hits,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "note": "Lexical candidates, not a factual answer or calibrated evidence sufficiency score",
        }

    def evidence(self, chunk_id: str) -> dict:
        with self.connect() as db:
            row = db.execute(
                "SELECT c.*,v.document_id,v.title,v.source_name,v.source_sha256,d.active_version,d.deleted,"
                "p.text AS page_text,p.char_start AS page_source_start,p.flags FROM chunks c "
                "JOIN versions v ON v.version_id=c.version_id JOIN documents d ON d.document_id=v.document_id "
                "JOIN pages p ON p.version_id=c.version_id AND p.page_number=c.page_number WHERE c.chunk_id=?",
                (chunk_id,),
            ).fetchone()
        if row is None:
            raise KBError("evidence_not_found", "Evidence ID not found", 404)
        if row["deleted"] or row["active_version"] != row["version_id"]:
            raise KBError(
                "evidence_unavailable",
                "Document was deleted or replaced; evidence is no longer active",
                410,
            )
        result = dict(row)
        for key in ("row_id", "active_version", "deleted"):
            result.pop(key)
        result["flags"] = json.loads(result["flags"])
        result["offset_unit"] = (
            "Unicode code points in UTF-8-sig decoded Markdown, before normalization"
        )
        result["page_number_kind"] = (
            "Markdown Page marker; PDF/printed pagination not independently verified"
        )
        assert (
            result["page_text"][result["page_char_start"] : result["page_char_end"]]
            == result["text"]
        )
        return result

    def active_chunks(self) -> list[dict]:
        """One consistent snapshot for building a derived vector index."""
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT c.chunk_id,c.version_id,c.text_sha256,c.text,v.document_id,v.title "
                    "FROM chunks c JOIN versions v ON v.version_id=c.version_id "
                    "JOIN documents d ON d.document_id=v.document_id AND d.active_version=c.version_id "
                    "WHERE d.deleted=0 ORDER BY c.chunk_id"
                )
            ]

    @staticmethod
    def fingerprint_chunks(rows: list[dict]) -> str:
        return digest(
            json.dumps(
                [
                    (r["document_id"], r["version_id"], r["chunk_id"], r["text_sha256"])
                    for r in rows
                ],
                separators=(",", ":"),
                sort_keys=True,
            )
        )

    def corpus_fingerprint(self, db=None) -> str:
        if db is None:
            with self.connect() as connection:
                return self.corpus_fingerprint(connection)
        rows = [
            dict(r)
            for r in db.execute(
                "SELECT v.document_id,c.version_id,c.chunk_id,c.text_sha256 FROM chunks c "
                "JOIN versions v ON v.version_id=c.version_id JOIN documents d "
                "ON d.active_version=c.version_id AND d.document_id=v.document_id "
                "WHERE d.deleted=0 ORDER BY c.chunk_id"
            )
        ]
        return self.fingerprint_chunks(rows)

    def active_hits(self, chunk_ids: list[str]) -> dict[str, dict]:
        if not chunk_ids:
            return {}
        if len(chunk_ids) > 100:
            raise ValueError("Too many candidate chunk IDs")
        with self.connect() as db:
            rows = db.execute(
                "SELECT c.chunk_id,v.document_id,c.version_id,v.title,c.page_number AS page_start,"
                "c.page_number AS page_end,c.char_start,c.char_end,c.text,c.text_sha256,p.flags "
                "FROM chunks c JOIN versions v ON v.version_id=c.version_id JOIN documents d "
                "ON d.active_version=c.version_id AND d.document_id=v.document_id JOIN pages p "
                "ON p.version_id=c.version_id AND p.page_number=c.page_number "
                "WHERE d.deleted=0 AND c.chunk_id IN (" + ",".join("?" for _ in chunk_ids) + ")",
                chunk_ids,
            ).fetchall()
        hits = {}
        for row in rows:
            hit = dict(row)
            hit["flags"] = json.loads(hit["flags"])
            hit["evidence_url"] = "/v1/evidence/" + hit["chunk_id"]
            hit["bm25_distance"] = None
            hits[hit["chunk_id"]] = hit
        return hits

    def stats(self) -> dict:
        docs = self.documents()
        return {
            "documents": len(docs),
            "pages": sum(d["page_count"] for d in docs),
            "chunks": sum(d["chunk_count"] for d in docs),
            "schema_version": 1,
            "chunk_recipe": asdict(self.recipe),
            "external_calls_enabled": False,
            "retrievers": ["sqlite_fts5_bm25"],
            "answer_generation_enabled": False,
        }
