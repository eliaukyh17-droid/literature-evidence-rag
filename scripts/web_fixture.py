"""Deterministic local browser fixture using self-authored example documents."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from literature_kb.api import create_app as application
from literature_kb.config import ROOT, Settings
from literature_kb.generation import PreparedPrompt
from literature_kb.store import Store


def create_app():
    folder = Path(os.environ["KB_WEB_TEST_DIR"]).resolve(strict=True)
    if not folder.is_relative_to((ROOT / ".local").resolve()):
        raise RuntimeError("Browser fixture storage must be private")
    settings = Settings(ROOT / "examples", folder / "index")
    store = Store(settings)
    store.import_document("guide.md")
    store.import_document("catalog.md")

    class Backend:
        demo_mode = True

        def __init__(self):
            self.question = ""
            self.state = {"calls": 0, "active": 0, "cancelled": 0}

        def record(self):
            (folder / "fixture_state.json").write_text(json.dumps(self.state), encoding="utf-8")

        async def prepare(self, messages):
            self.question = json.loads(messages[1]["content"])["question"]
            return PreparedPrompt("browser-fixture", 100, 2048)

        async def generate(self, prepared, schema, max_tokens, *, on_delta, on_progress):
            self.state["calls"] += 1
            self.state["active"] += 1
            self.record()
            content = json.dumps(
                {
                    "insufficient_evidence": False,
                    "answer": "证据定位需要保留文档编号、版本编号及相对原文的字符位置，以便回查对应片段。[S1]",
                    "citations": ["S1"],
                },
                ensure_ascii=False,
            )
            if "安全测试" in self.question:
                content = content.replace(
                    "[S1]", '<img src=x onerror=\\"window.injected=true\\">[S1]'
                )
            if "错误测试" in self.question:
                content = content.replace("S1", "S99")
            try:
                await on_progress({"total": 100, "cache": 0, "processed": 50})
                for offset in range(0, len(content), 5):
                    await on_delta(content[offset : offset + 5])
                    if "停止测试" in self.question and offset >= 60:
                        await asyncio.sleep(30)
                    await asyncio.sleep(0.04)
                return {
                    "content": content,
                    "model": "browser-fixture",
                    "finish_reason": "eos",
                    "completion_tokens": 40,
                }
            except asyncio.CancelledError:
                self.state["cancelled"] += 1
                raise
            finally:
                self.state["active"] -= 1
                self.record()

        async def aclose(self):
            pass

    return application(settings, generation_factory=lambda _: Backend())
