from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    source_root: Path
    data_dir: Path
    max_source_bytes: int = 20 * 1024 * 1024
    chunk_chars: int = 900
    overlap_chars: int = 120
    embedding_model_dir: Path | None = None
    embedding_threads: int = 2
    chunk_max_tokens: int | None = None
    local_llm_url: str | None = None
    local_llm_model: str = "literature-qwen3-local"
    generation_context_tokens: int = 3072
    answer_timeout_seconds: float = 120

    @classmethod
    def from_env(cls) -> Settings:
        # Never load parent .env files, CSVs, keys, budgets or customer ledgers.
        return cls(
            source_root=Path(os.environ.get("KB_SOURCE_ROOT", ROOT / "examples")),
            data_dir=Path(os.environ.get("KB_DATA_DIR", ROOT / ".local/demo")),
            embedding_model_dir=Path(os.environ["KB_EMBEDDING_MODEL_DIR"])
            if os.environ.get("KB_EMBEDDING_MODEL_DIR")
            else None,
            chunk_max_tokens=int(os.environ["KB_CHUNK_MAX_TOKENS"])
            if os.environ.get("KB_CHUNK_MAX_TOKENS")
            else None,
            local_llm_url=os.environ.get("KB_LOCAL_LLM_URL") or None,
            local_llm_model=os.environ.get("KB_LOCAL_LLM_MODEL", "literature-qwen3-local"),
        )
