from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath

from .errors import KBError

PARSER_VERSION = "page-exact-offsets-v1"
PAGE = re.compile(r"^## Page (\d+)[ \t]*\r?$", re.MULTILINE)
SOURCE = re.compile(r"^> Source: `([^`]+)`[ \t]*\r?$", re.MULTILINE)


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def safe_source(root: Path, relative_path: str, max_bytes: int) -> Path:
    # Reject both Windows and POSIX absolute paths, traversal and NTFS alternate streams.
    portable = relative_path.replace("\\", "/")
    rel = Path(portable)
    if (
        not portable
        or rel.is_absolute()
        or rel.anchor
        or portable.startswith("/")
        or PureWindowsPath(portable).drive
        or any(p in {"..", "."} for p in portable.split("/"))
        or ":" in portable
        or rel.suffix.lower() != ".md"
    ):
        raise KBError(
            "invalid_source", "Only relative Markdown paths inside the source root are allowed"
        )
    root = root.resolve(strict=True)
    candidate = root / rel
    current = root
    for component in rel.parts:
        current = current / component
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            raise KBError("invalid_source", "Linked sources are not allowed")
    try:
        candidate = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise KBError("source_not_found", "Markdown source does not exist", 404) from exc
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise KBError("invalid_source", "Source must be a regular file inside the source root")
    if candidate.stat().st_size > max_bytes:
        raise KBError("source_too_large", "Source exceeds the configured byte limit", 413)
    return candidate


@dataclass(frozen=True)
class ChunkRecipe:
    max_chars: int = 900
    overlap: int = 120
    min_page_chars: int = 30
    parser: str = PARSER_VERSION
    max_tokens: int | None = None
    tokenizer_sha256: str | None = None

    def __post_init__(self):
        if not 100 <= self.max_chars <= 4000 or not 0 <= self.overlap < self.max_chars // 2:
            raise ValueError("Invalid chunk size/overlap")
        if not 1 <= self.min_page_chars <= self.max_chars:
            raise ValueError("Invalid minimum page size")
        if self.max_tokens is not None and (
            not 32 <= self.max_tokens <= 512 or not self.tokenizer_sha256
        ):
            raise ValueError("A token budget of 32-512 requires a pinned tokenizer")

    @property
    def fingerprint(self) -> str:
        fields = asdict(self)
        if self.max_tokens is None:
            # Preserve the original character-only recipe fingerprint.
            fields.pop("max_tokens")
            fields.pop("tokenizer_sha256")
        return digest(json.dumps(fields, sort_keys=True))


@dataclass(frozen=True)
class Page:
    number: int
    char_start: int
    text: str
    flags: tuple[str, ...]


@dataclass(frozen=True)
class Chunk:
    page: int
    page_char_start: int
    page_char_end: int
    char_start: int
    char_end: int
    text: str
    text_sha256: str


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    source_name: str | None
    pages: list[Page]
    chunks: list[Chunk]
    skipped_pages: list[int]


def parse_document(text: str, recipe: ChunkRecipe, counter=None) -> ParsedDocument:
    if recipe.max_tokens is not None and (
        counter is None or counter.sha256 != recipe.tokenizer_sha256
    ):
        raise ValueError("The configured token recipe requires its matching counter")
    markers = list(PAGE.finditer(text))
    numbers = [int(m[1]) for m in markers]
    if not numbers or numbers != list(range(1, len(numbers) + 1)):
        raise KBError(
            "invalid_page_markers", "Expected unique sequential Page markers starting at 1"
        )
    title = next(
        (s[2:].strip() for s in text[: markers[0].start()].splitlines() if s.startswith("# ")),
        "Untitled",
    )
    title = title[:500]
    source = SOURCE.search(text[: markers[0].start()])
    pages, chunks, skipped = [], [], []
    for i, marker in enumerate(markers):
        start = marker.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        body = text[start:end]
        flags = []
        if len(re.sub(r"\s", "", body)) < 80:
            flags.append("sparse")
        if "\ufffd" in body:
            flags.append("replacement_character")
        if re.search(r"\[OCR_(?:REQUIRED|FAILED)\b", body):
            flags.append("ocr_unavailable")
        page = Page(int(marker[1]), start, body, tuple(flags))
        pages.append(page)
        if "ocr_unavailable" in flags or len(re.sub(r"\s", "", body)) < recipe.min_page_chars:
            skipped.append(page.number)
            continue
        cursor = 0
        while cursor < len(body):
            stop = min(cursor + recipe.max_chars, len(body))
            if stop < len(body):
                # Prefer paragraph/line/sentence boundaries in the latter half of the window.
                floor = cursor + recipe.max_chars // 2
                boundaries = list(re.finditer(r"\n\s*\n|[。！？.!?]\s|\n", body[floor:stop]))
                if boundaries:
                    stop = floor + boundaries[-1].end()
            left, right = cursor, stop
            while left < right and body[left].isspace():
                left += 1
            while right > left and body[right - 1].isspace():
                right -= 1
            if left < right:
                part = body[left:right]
                chunks.append(
                    Chunk(page.number, left, right, start + left, start + right, part, digest(part))
                )
            if stop == len(body):
                break
            cursor = max(cursor + 1, stop - recipe.overlap)
    if recipe.max_tokens is not None:
        from .token_budget import split_to_budget

        bounded = []
        for c in chunks:
            for left, right in split_to_budget(
                c.text, title, counter, recipe.max_tokens, recipe.overlap
            ):
                part = c.text[left:right]
                bounded.append(
                    Chunk(
                        c.page,
                        c.page_char_start + left,
                        c.page_char_start + right,
                        c.char_start + left,
                        c.char_start + right,
                        part,
                        digest(part),
                    )
                )
        chunks = bounded
    if not chunks:
        raise KBError(
            "no_indexable_content", "No indexable page content; original version remains active"
        )
    # Snapshot offsets address exact decoded Markdown, not normalized/cleaned text or PDF bytes.
    return ParsedDocument(title[:500], source[1] if source else None, pages, chunks, skipped)
