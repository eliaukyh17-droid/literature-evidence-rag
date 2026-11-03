"""Exercise real loopback HTTP using only synthetic fixtures; stop owned child on exit."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from literature_kb.config import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path, help="Also exercise real local embeddings on synthetic documents"
    )
    args = parser.parse_args()
    run_dir = ROOT / ".local" / ("http-smoke-" + uuid.uuid4().hex[:12])
    run_dir.mkdir(parents=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {
        **os.environ,
        "KB_SOURCE_ROOT": str(ROOT / "examples"),
        "KB_DATA_DIR": str(run_dir / "store"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    env.pop("KB_EMBEDDING_MODEL_DIR", None)
    if args.model_dir:
        from literature_kb.config import Settings
        from literature_kb.retrieval import RetrievalService
        from literature_kb.store import Store

        store = Store(
            Settings(ROOT / "examples", run_dir / "store", embedding_model_dir=args.model_dir)
        )
        for source in ("guide.md", "catalog.md"):
            store.import_document(source)
        service = RetrievalService.configured(store)
        try:
            service.dense.build()
        finally:
            service.close()
        del service
        env["KB_EMBEDDING_MODEL_DIR"] = str(args.model_dir.resolve())
    # Bypass corporate/system proxies for these strictly local requests.
    client = build_opener(ProxyHandler({}))

    def request(route, data=None, method=None):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8") if data is not None else None
        req = Request(
            f"http://127.0.0.1:{port}{route}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json", "X-Request-ID": "http_smoke"},
        )
        # A cold ONNX session may take several seconds on a memory-constrained laptop.
        with client.open(req, timeout=15) as response:
            raw = response.read()
            return json.loads(raw) if raw else None

    child = None
    started = time.perf_counter()
    with (
        (run_dir / "server.stdout.log").open("wb") as stdout,
        (run_dir / "server.stderr.log").open("wb") as stderr,
    ):
        try:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "literature_kb.api:create_app",
                    "--factory",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--no-access-log",
                    "--log-config",
                    "logging.json",
                ],
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
            )
            deadline = time.monotonic() + 20
            while True:
                if child.poll() is not None:
                    raise RuntimeError(
                        "HTTP server exited before readiness; inspect owned server log"
                    )
                try:
                    health = request("/v1/health")
                    break
                except (URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("HTTP server readiness timeout") from None
                    time.sleep(0.2)
            assert health["external_calls_enabled"] is False
            imported = [
                request("/v1/documents/import", {"relative_path": p})
                for p in ("guide.md", "catalog.md")
            ]
            hit = request("/v1/search", {"query": "证据定位"})["hits"][0]
            evidence = request(hit["evidence_url"])
            assert (
                evidence["page_text"][evidence["page_char_start"] : evidence["page_char_end"]]
                == hit["text"]
            )
            alias_hit = request("/v1/search", {"query": "莫诺马赫", "expand_aliases": True})[
                "hits"
            ][0]
            assert alias_hit["document_id"] == imported[1]["document_id"]
            if args.model_dir:
                for mode in ("dense", "hybrid", "hybrid_terms"):
                    response = request(
                        "/v1/search",
                        {
                            "query": "莫诺马赫" if mode == "hybrid_terms" else "文档版本与证据",
                            "mode": mode,
                        },
                    )
                    assert response["hits"] and response["metadata"]["embedding_id"]
                    assert request(response["hits"][0]["evidence_url"])["text"]
                    assert response["hits"][0]["dense_score"] is not None
                    if mode == "hybrid_terms":
                        assert response["metadata"]["query_terms"][0]["source"] == "莫诺马赫"
            request("/v1/documents/" + imported[0]["document_id"], method="DELETE")
            try:
                request(hit["evidence_url"])
                raise AssertionError("Deleted evidence remained accessible")
            except HTTPError as exc:
                assert exc.code == 410
            assert "/v1/search" in request("/openapi.json")["paths"]
            if args.model_dir:
                try:
                    request("/v1/search", {"query": "evidence", "mode": "dense"})
                    raise AssertionError("Stale dense index remained accessible after deletion")
                except HTTPError as exc:
                    assert exc.code == 503
            report = {
                "passed": True,
                "transport": "real loopback HTTP",
                "source": "synthetic fixtures only",
                "verified": [
                    "health",
                    "import",
                    "search",
                    "evidence offsets",
                    "alias expansion",
                    "delete",
                    "410",
                    "OpenAPI",
                ],
                "api_calls": 0,
                "port": port,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        finally:
            if child is not None:
                if os.name == "nt" and child.poll() is None:
                    subprocess.run(
                        ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                elif child.poll() is None:
                    child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
    with socket.socket() as probe:
        probe.settimeout(1)
        port_closed = probe.connect_ex(("127.0.0.1", port)) != 0
    report["owned_server_stopped"] = child.poll() is not None and port_closed
    assert report["owned_server_stopped"], "Owned HTTP server did not stop"
    server_log = (run_dir / "server.stderr.log").read_text(encoding="utf-8", errors="replace")
    assert "request_id=http_smoke" in server_log, "Request tracing was not configured"
    assert "证据定位" not in server_log and "莫诺马赫" not in server_log, (
        "Query text leaked to logs"
    )
    report["request_trace_verified"] = True
    report["real_local_embedding_tested"] = args.model_dir is not None
    (run_dir / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({**report, "output": str(run_dir / "result.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
