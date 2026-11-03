from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import anyio
import httpx
import pytest
from fastapi.testclient import TestClient

from literature_kb.api import create_app
from literature_kb.errors import KBError
from literature_kb.generation import LocalLlamaBackend, PreparedPrompt
from literature_kb.streaming import answer_prefix


@pytest.mark.parametrize("escaped", [True, False])
def test_partial_json_answer_is_monotonic_across_every_character(escaped):
    answer = '中文，русский，😀，引号"与反斜杠\\\n换行。[S1]'
    raw = json.dumps(
        {"insufficient_evidence": False, "answer": answer, "citations": ["S1"]},
        ensure_ascii=escaped,
    )
    previous = ""
    for cut in range(len(raw) + 1):
        text = answer_prefix(raw[:cut])
        assert text.startswith(previous)
        assert answer.startswith(text)
        text.encode("utf-8", errors="strict")
        previous = text
    assert previous == answer


def test_answer_key_inside_another_value_is_not_mistaken_for_a_field():
    raw = json.dumps({"other": 'answer ": text', "answer": "正确"}, ensure_ascii=False)
    assert answer_prefix(raw[:-2]) == "正确"


class ModelStream(httpx.AsyncByteStream):
    def __init__(self, content):
        self.content, self.closed = content, False

    async def __aiter__(self):
        for offset in range(0, len(self.content), 3):
            yield self.content[offset : offset + 3]

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("finish", ["eos", "limit", None])
def test_native_stream_reads_split_utf8_and_requires_normal_terminal_event(finish):
    raw = '{"insufficient_evidence":false,"answer":"测试😀[S1]","citations":["S1"]}'
    packets = [{"prompt_progress": {"total": 20, "cache": 0, "processed": 10}}]
    packets += [{"content": piece, "stop": False} for piece in (raw[:35], raw[35:])]
    if finish:
        packets.append(
            {"content": "", "stop": True, "stop_type": finish, "timings": {"predicted_n": 20}}
        )
    body = (
        ": ping\r\n\r\n"
        + "".join("data: " + json.dumps(p, ensure_ascii=False) + "\r\n\r\n" for p in packets)
    ).encode()
    stream = ModelStream(body)
    received, progress = [], []

    def handler(request):
        payload = json.loads(request.content)
        assert payload["stream"] is True and payload["return_progress"] is True
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=stream)

    async def on_delta(value):
        received.append(value)

    async def on_progress(value):
        progress.append(value)

    async def run():
        backend = LocalLlamaBackend(
            "http://127.0.0.1:19999", "fixture", 2048, transport=httpx.MockTransport(handler)
        )
        try:
            if finish == "eos":
                result = await backend.generate(
                    PreparedPrompt("fixture", 10, 2048),
                    {},
                    128,
                    on_delta=on_delta,
                    on_progress=on_progress,
                )
                assert result["content"] == raw
            else:
                with pytest.raises(KBError) as error:
                    await backend.generate(
                        PreparedPrompt("fixture", 10, 2048),
                        {},
                        128,
                        on_delta=on_delta,
                        on_progress=on_progress,
                    )
                assert error.value.code == "generation_incomplete"
            assert "".join(received) == raw and progress[0]["processed"] == 10
            assert stream.closed
        finally:
            await backend.aclose()

    asyncio.run(run())


def test_cancel_scope_waits_for_upstream_socket_to_close():
    class SlowCloseStream(ModelStream):
        async def aclose(self):
            await anyio.sleep(0.01)
            self.closed = True

    stream = SlowCloseStream(b'data: {"content":"partial","stop":false}\n\n')

    def handler(request):
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=stream)

    async def run():
        backend = LocalLlamaBackend(
            "http://127.0.0.1:19999", "fixture", 2048, transport=httpx.MockTransport(handler)
        )
        try:
            with anyio.CancelScope() as scope:

                async def on_delta(piece):
                    scope.cancel()
                    await anyio.sleep_forever()

                await backend.generate(
                    PreparedPrompt("fixture", 10, 2048), {}, 128, on_delta=on_delta
                )
            assert stream.closed
            assert not backend._lock.locked()
        finally:
            await backend.aclose()

    anyio.run(run)


class StreamingFixture:
    def __init__(self):
        self.calls = 0
        self.content = (
            '{"insufficient_evidence":false,"answer":"夹具结论。[S1]","citations":["S1"]}'
        )
        self.delay = 0
        self.cancelled = False

    async def prepare(self, messages):
        return PreparedPrompt("fixture", 100, 2048)

    async def generate(self, prepared, schema, max_tokens, *, on_delta, on_progress):
        self.calls += 1
        try:
            await on_progress({"total": 100, "cache": 0, "processed": 50})
            for offset in range(0, len(self.content), 4):
                await on_delta(self.content[offset : offset + 4])
                await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return {
            "content": self.content,
            "model": "fixture",
            "finish_reason": "eos",
            "completion_tokens": 20,
        }

    async def aclose(self):
        pass


def parsed_events(text):
    result = []
    for frame in text.split("\n\n"):
        lines = frame.splitlines()
        data = next((line[6:] for line in lines if line.startswith("data: ")), None)
        if data:
            event = next(line[7:] for line in lines if line.startswith("event: "))
            payload = json.loads(data)
            assert f"id: {payload['request_id']}:{payload['sequence']}" in lines
            result.append((event, payload))
    assert [payload["sequence"] for _, payload in result] == list(range(1, len(result) + 1))
    return result


def test_sse_http_contract_final_citations_and_static_ui(settings, write_source):
    write_source()
    backend = StreamingFixture()
    with TestClient(create_app(settings, generation_factory=lambda _: backend)) as client:
        client.post("/v1/documents/import", json={"relative_path": "test.md"})
        response = client.post("/v1/answer/stream", json={"query": "证据定位", "mode": "bm25"})
        assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
        events = parsed_events(response.text)
        assert events[0][0] == "started" and events[-1][0] == "done"
        assert all(data["request_id"] == response.headers["X-Request-ID"] for _, data in events)
        assert (
            "".join(data["text"] for event, data in events if event == "answer_delta")
            == "夹具结论。[S1]"
        )
        done = events[-1][1]
        assert (
            done["status"] == "answered" and done["metadata"]["first_answer_delta_ms"] is not None
        )
        assert client.get(done["citations"][0]["evidence_url"]).status_code == 200
        assert client.get("/").status_code == 200
        assert "script-src 'self'" in client.get("/").headers["content-security-policy"]
        assert client.get("/assets/app.js").status_code == 200
        assert client.get("/assets/style.css").status_code == 200
        content = client.get("/openapi.json").json()["paths"]["/v1/answer/stream"]["post"][
            "responses"
        ]["200"]["content"]
        assert set(content) == {"text/event-stream"}
        assert (
            client.post(
                "/v1/answer/stream", headers={"Last-Event-ID": "old:1"}, json={"query": "test"}
            ).status_code
            == 400
        )


def test_invalid_stream_has_error_after_preview_and_can_be_retried(settings, write_source):
    write_source()
    backend = StreamingFixture()
    backend.content = backend.content.replace("S1", "S99")
    with TestClient(create_app(settings, generation_factory=lambda _: backend)) as client:
        client.post("/v1/documents/import", json={"relative_path": "test.md"})
        events = parsed_events(
            client.post("/v1/answer/stream", json={"query": "证据定位", "mode": "bm25"}).text
        )
        assert any(event == "answer_delta" for event, _ in events)
        assert events[-1][0] == "error" and events[-1][1]["code"] == "invalid_citations"
        assert not any(event == "done" for event, _ in events)
        backend.content = backend.content.replace("S99", "S1")
        assert (
            parsed_events(
                client.post("/v1/answer/stream", json={"query": "证据定位", "mode": "bm25"}).text
            )[-1][0]
            == "done"
        )
        assert backend.calls == 2


def test_stream_timeout_cancels_generation_and_has_one_terminal_event(settings, write_source):
    write_source()
    backend = StreamingFixture()
    backend.delay = 0.1
    config = replace(settings, answer_timeout_seconds=0.03)
    with TestClient(create_app(config, generation_factory=lambda _: backend)) as client:
        client.post("/v1/documents/import", json={"relative_path": "test.md"})
        events = parsed_events(
            client.post("/v1/answer/stream", json={"query": "证据定位", "mode": "bm25"}).text
        )
        assert events[-1][0] == "error" and events[-1][1]["code"] == "answer_timeout"
        assert backend.cancelled and backend.calls == 1
