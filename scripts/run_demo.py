"""Start the public, deterministic interaction demo with self-authored examples."""

from __future__ import annotations

import argparse
import asyncio
import json

from literature_kb.api import create_app as application
from literature_kb.config import ROOT, Settings
from literature_kb.generation import PreparedPrompt
from literature_kb.store import Store


class DemoBackend:
    demo_mode = True

    async def prepare(self, messages):
        self.evidence = json.loads(messages[1]["content"])["evidence"]
        return PreparedPrompt("public-demo", 100, 2048)

    async def generate(self, prepared, schema, max_tokens, *, on_delta=None, on_progress=None):
        source = self.evidence[0]
        content = json.dumps(
            {
                "insufficient_evidence": False,
                "answer": "这是用于体验界面的示例回答。检索到的片段为：\n"
                + source["text"][:160]
                + "\n["
                + source["source_id"]
                + "]",
                "citations": [source["source_id"]],
            },
            ensure_ascii=False,
        )
        if on_progress:
            await on_progress({"total": 100, "cache": 0, "processed": 100})
        if on_delta:
            for offset in range(0, len(content), 4):
                await on_delta(content[offset : offset + 4])
                await asyncio.sleep(0.035)
        return {
            "content": content,
            "model": "deterministic-public-demo",
            "finish_reason": "eos",
            "completion_tokens": None,
        }

    async def aclose(self):
        pass


def create_app():
    settings = Settings(ROOT / "examples", ROOT / ".local/public-demo")
    store = Store(settings)
    for name in ("guide.md", "catalog.md"):
        store.import_document(name)
    return application(settings, generation_factory=lambda _: DemoBackend())


def main():
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(
        "Public interaction demo with deterministic sample answers: http://127.0.0.1:"
        + str(args.port),
        flush=True,
    )
    uvicorn.run(
        create_app(),
        host="127.0.0.1",
        port=args.port,
        access_log=False,
        log_config=str(ROOT / "logging.json"),
    )


if __name__ == "__main__":
    main()
