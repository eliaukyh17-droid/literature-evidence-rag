"""Strictly loopback llama.cpp adapter. No keys, external hosts or automatic retries."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from urllib.parse import urlsplit

import anyio
import httpx

from .errors import KBError


@dataclass(frozen=True)
class PreparedPrompt:
    text: str
    tokens: int
    context_limit: int


class LocalLlamaBackend:
    def __init__(
        self,
        base_url: str,
        model: str,
        context_limit: int,
        *,
        transport=None,
        response_observer=None,
    ):
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "::1"}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError(
                "Local generation URL must be a literal loopback HTTP origin without credentials"
            )
        if not 512 <= context_limit <= 16384:
            raise ValueError("Invalid generation context limit")
        self.model, self.context_limit = model, context_limit
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            trust_env=False,
            follow_redirects=False,
            transport=transport,
            timeout=httpx.Timeout(90, connect=3, pool=3),
            limits=httpx.Limits(max_connections=2),
        )
        self.actual_context = None
        self.response_observer = response_observer
        self._lock = asyncio.Lock()

    async def aclose(self):
        await self.client.aclose()

    async def _request(self, method: str, path: str, payload=None):
        try:
            response = await self.client.request(method, path, json=payload)
        except httpx.TimeoutException as exc:
            raise KBError("generation_timeout", "Local model request timed out", 504) from exc
        except httpx.TransportError as exc:
            raise KBError(
                "generation_unavailable", "Local model server is unavailable", 503
            ) from exc
        if response.status_code == 503:
            raise KBError("generation_busy", "Local model server is not ready or is busy", 503)
        if response.status_code != 200:
            raise KBError(
                "generation_protocol_error", "Local model returned an unexpected status", 502
            )
        if len(response.content) > 256 * 1024:
            raise KBError(
                "generation_protocol_error", "Local model response exceeded the byte limit", 502
            )
        try:
            return response.json()
        except (ValueError, UnicodeError) as exc:
            raise KBError(
                "generation_protocol_error", "Local model returned invalid JSON", 502
            ) from exc

    async def _identity(self):
        # Verify BEFORE sending any question/evidence. Recheck on each answer preparation.
        models = await self._request("GET", "/v1/models")
        if self.model not in {r.get("id") for r in models.get("data", [])}:
            raise KBError(
                "generation_model_mismatch",
                "Local server model alias differs from configuration",
                503,
            )
        props = await self._request("GET", "/props")
        context = props.get("default_generation_settings", {}).get("n_ctx")
        if not isinstance(context, int) or context < 512:
            raise KBError("generation_protocol_error", "Server context size is unavailable", 502)
        self.actual_context = min(context, self.context_limit)

    async def prepare(self, messages: list[dict]) -> PreparedPrompt:
        await self._identity()
        formatted = await self._request(
            "POST",
            "/apply-template",
            {"messages": messages, "chat_template_kwargs": {"enable_thinking": False}},
        )
        prompt = formatted.get("prompt")
        if not isinstance(prompt, str) or len(prompt) > 80_000:
            raise KBError("generation_protocol_error", "Invalid formatted model prompt", 502)
        tokenized = await self._request(
            "POST", "/tokenize", {"content": prompt, "add_special": True, "parse_special": True}
        )
        ids = tokenized.get("tokens")
        if not isinstance(ids, list) or any(not isinstance(i, int) for i in ids):
            raise KBError("generation_protocol_error", "Invalid model token count", 502)
        return PreparedPrompt(prompt, len(ids), self.actual_context)

    async def _stream(self, payload, on_delta, on_progress):
        chunks, data_lines = [], []
        wire_bytes = 0
        try:
            response = await self.client.send(
                self.client.build_request("POST", "/completion", json=payload), stream=True
            )
            try:
                if response.status_code != 200:
                    status = 503 if response.status_code == 503 else 502
                    raise KBError("generation_unavailable", "Local stream could not start", status)
                if "text/event-stream" not in response.headers.get("content-type", ""):
                    raise KBError("generation_protocol_error", "Expected a model SSE stream", 502)
                async for line in response.aiter_lines():
                    wire_bytes += len(line.encode("utf-8"))
                    if wire_bytes > 1024 * 1024:
                        raise KBError(
                            "generation_protocol_error", "Model stream exceeded limit", 502
                        )
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip(" "))
                    if line or not data_lines:
                        continue
                    data = "\n".join(data_lines)
                    data_lines.clear()
                    if data == "[DONE]":
                        break
                    try:
                        packet = json.loads(data)
                    except ValueError as exc:
                        raise KBError(
                            "generation_protocol_error", "Invalid model stream JSON", 502
                        ) from exc
                    if not isinstance(packet, dict) or "error" in packet:
                        raise KBError(
                            "generation_protocol_error", "Invalid model stream event", 502
                        )
                    progress = packet.get("prompt_progress")
                    if isinstance(progress, dict) and on_progress is not None:
                        await on_progress(
                            {key: progress.get(key) for key in ("total", "cache", "processed")}
                        )
                    piece = packet.get("content", "")
                    if not isinstance(piece, str):
                        raise KBError("generation_protocol_error", "Invalid model text delta", 502)
                    if piece:
                        chunks.append(piece)
                        await on_delta(piece)
                    if packet.get("stop") is True:
                        return {**packet, "content": "".join(chunks)}
            finally:
                # Disconnect cancellation must not interrupt closing the upstream socket.
                with anyio.CancelScope(shield=True):
                    await response.aclose()
        except httpx.TimeoutException as exc:
            raise KBError("generation_timeout", "Local model stream timed out", 504) from exc
        except httpx.TransportError as exc:
            raise KBError("generation_unavailable", "Local model stream disconnected", 503) from exc
        raise KBError("generation_incomplete", "Model stream ended without a completion event", 502)

    async def generate(
        self,
        prepared: PreparedPrompt,
        schema: dict,
        max_tokens: int,
        *,
        on_delta=None,
        on_progress=None,
    ):
        if self._lock.locked():
            raise KBError("generation_busy", "Another local model answer is in progress", 503)
        payload = {
            "prompt": prepared.text,
            "n_predict": max_tokens,
            "temperature": 0.2,
            "seed": 42,
            "stream": on_delta is not None,
            "json_schema": schema,
            "cache_prompt": True,
        }
        async with self._lock:
            if on_delta is None:
                result = await self._request("POST", "/completion", payload)
            else:
                result = await self._stream(
                    {**payload, "return_progress": True, "sse_ping_interval": 10},
                    on_delta,
                    on_progress,
                )
        if self.response_observer is not None:
            self.response_observer(result)
        if result.get("truncated") or result.get("stop_type") == "limit":
            raise KBError(
                "generation_incomplete",
                "Model output reached a token/context limit; no partial answer returned",
                502,
            )
        if not isinstance(result.get("content"), str) or result.get("stop_type") not in {
            "eos",
            "word",
        }:
            raise KBError(
                "generation_protocol_error", "Model completion did not finish normally", 502
            )
        return {
            "content": result["content"],
            "model": self.model,
            "prompt_tokens": result.get("tokens_evaluated", prepared.tokens),
            "completion_tokens": result.get("timings", {}).get("predicted_n"),
            "finish_reason": result["stop_type"],
        }
