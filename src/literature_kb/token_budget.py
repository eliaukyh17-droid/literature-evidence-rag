from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path

from .errors import KBError
from .text import prepare_text


class E5TokenBudget:
    """Count the EXACT E5 input, including title, prefix and special tokens."""

    def __init__(self, model_dir: Path):
        from tokenizers import Tokenizer

        path = model_dir / "tokenizer.json"
        receipt = json.loads((model_dir / "download_manifest.json").read_text(encoding="utf-8"))
        with path.open("rb") as stream:
            self.sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        if self.sha256 != receipt["files"]["tokenizer.json"]["sha256"]:
            raise ValueError("Tokenizer checksum mismatch")
        self.tokenizer = Tokenizer.from_file(str(path))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.lock = threading.RLock()

    def count(self, title: str, text: str) -> int:
        with self.lock:
            return len(self.tokenizer.encode("passage: " + prepare_text(title + "\n" + text)).ids)


def split_to_budget(text: str, title: str, counter, limit: int, overlap: int):
    """Return raw character spans; keep already fitting chunks exactly unchanged.

    Binary search is only a proposal: tokenizer lengths need not be monotonic.
    Each proposed span is explicitly checked and shrunk before it is emitted.
    """
    if counter.count(title, "") >= limit:
        raise KBError(
            "title_exceeds_token_budget", "Document title leaves no token budget for evidence"
        )
    if counter.count(title, text) <= limit:
        return [(0, len(text))]
    spans, cursor = [], 0
    while cursor < len(text):
        if counter.count(title, text[cursor:]) <= limit:
            stop = len(text)
        else:
            lo, hi, stop = cursor + 1, len(text), cursor
            while lo <= hi:
                middle = (lo + hi) // 2
                if counter.count(title, text[cursor:middle]) <= limit:
                    stop, lo = middle, middle + 1
                else:
                    hi = middle - 1
            if stop == cursor:
                raise KBError(
                    "token_budget_too_small", "A source character cannot fit the token budget"
                )
            floor = cursor + (stop - cursor) * 2 // 3
            boundaries = list(re.finditer(r"[。！？.!?]\s|\n|\s", text[floor:stop]))
            if boundaries:
                stop = floor + boundaries[-1].end()
        while stop > cursor and counter.count(title, text[cursor:stop]) > limit:
            stop -= 1
        if stop <= cursor:
            raise KBError("token_budget_too_small", "Source cannot fit the configured input window")
        left, right = cursor, stop
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            # Trimming can change tokenization. Recheck the actual emitted input.
            while right > left and counter.count(title, text[left:right]) > limit:
                right -= 1
            if right == left:
                raise KBError("token_budget_too_small", "Trimmed evidence cannot fit")
            spans.append((left, right))
            stop = min(stop, right)  # Never skip a tail removed during the final check.
        if stop == len(text):
            break
        cursor = max(cursor + 1, stop - min(overlap, max(0, (stop - cursor) // 4)))
    return spans
