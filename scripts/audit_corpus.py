"""Offline, read-only corpus inventory for the personal RAG extension.

This measures document/page-marker coverage, not OCR accuracy or retrieval quality.
No source text is copied, no PDFs are uploaded, and no API key is read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

PROJECT = Path(__file__).resolve().parents[2]
PAGE = re.compile(r"^## Page (\d+)\s*$", re.MULTILINE)
SOURCE = re.compile(r"^> Source: `([^`]+)`\s*$", re.MULTILINE)
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
CYRILLIC = re.compile(r"[\u0400-\u052f]")
OCR_MARKER = re.compile(r"\[OCR_(?:REQUIRED|FAILED)\b")


def normalized_name(name: str) -> str:
    return unicodedata.normalize("NFC", name.replace("\\", "/").rsplit("/", 1)[-1]).casefold().strip()


def inspect_markdown(text: str) -> dict:
    markers = list(PAGE.finditer(text))
    numbers = [int(m[1]) for m in markers]
    counts = Counter(numbers)
    gaps = sorted(set(range(1, max(numbers, default=0) + 1)) - set(numbers))
    sparse = 0
    nonempty = 0
    for i, marker in enumerate(markers):
        body = text[marker.end():markers[i+1].start() if i+1 < len(markers) else len(text)]
        chars = len(re.sub(r"\s", "", body))
        nonempty += chars > 0
        sparse += chars < 80
    han = len(HAN.findall(text))
    cyrillic = len(CYRILLIC.findall(text))
    if han >= 100 and cyrillic >= 100 and min(han, cyrillic) / max(han, cyrillic) >= .1:
        language = "mixed_han_cyrillic"
    elif cyrillic > han:
        language = "cyrillic_dominant"
    elif han > 0:
        language = "han_dominant"
    else:
        language = "other_or_sparse"
    source = SOURCE.search(text)
    return dict(source_header=source[1] if source else None,
                page_markers=len(markers), unique_page_markers=len(counts),
                duplicate_page_numbers=sorted(n for n, count in counts.items() if count > 1),
                out_of_order_page_numbers=numbers != sorted(numbers),
                nonpositive_page_numbers=sorted(n for n in numbers if n <= 0),
                missing_page_numbers=gaps, nonempty_pages=nonempty,
                sparse_pages_under_80_nonwhitespace_chars=sparse,
                explicit_ocr_failure_or_required_markers=len(OCR_MARKER.findall(text)),
                replacement_characters=text.count("\ufffd"),
                han_characters=han, cyrillic_characters=cyrillic,
                script_classification=language)


def inventory(md_root: Path, pdf_root: Path) -> dict:
    md_root, pdf_root = md_root.resolve(strict=True), pdf_root.resolve(strict=True)
    pdfs = sorted(pdf_root.rglob("*.pdf"))
    by_name = {}
    for path in pdfs:
        by_name.setdefault(normalized_name(path.name), []).append(path)
    records = []
    matched = set()
    for path in sorted(md_root.rglob("*.md")):
        if path.is_symlink() or not path.resolve().is_relative_to(md_root):
            raise ValueError("Unexpected Markdown source path")
        raw = path.read_bytes()
        report = inspect_markdown(raw.decode("utf-8-sig", errors="strict"))
        matches = by_name.get(normalized_name(report["source_header"]), []) if report["source_header"] else []
        # A header can be ambiguous; do not silently declare all candidates covered.
        if len(matches) == 1:
            matched.add(matches[0])
        records.append(dict(markdown=path.relative_to(md_root).as_posix(),
                            markdown_sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                            pdf_matches=[p.relative_to(pdf_root).as_posix() for p in matches],
                            pdf_mapping="unique" if len(matches) == 1 else "ambiguous" if matches else "unmatched",
                            **report))
    digest_counts = Counter(r["markdown_sha256"] for r in records)
    summary = dict(markdown_files=len(records), markdown_bytes=sum(r["bytes"] for r in records),
                   original_pdf_files=len(pdfs), page_markers=sum(r["page_markers"] for r in records),
                   sparse_pages_under_80_nonwhitespace_chars=sum(r["sparse_pages_under_80_nonwhitespace_chars"] for r in records),
                   explicit_ocr_failure_or_required_markers=sum(r["explicit_ocr_failure_or_required_markers"] for r in records),
                   replacement_characters=sum(r["replacement_characters"] for r in records),
                   documents_without_page_markers=sum(r["page_markers"] == 0 for r in records),
                   documents_with_numbering_flags=sum(bool(r["missing_page_numbers"] or r["duplicate_page_numbers"] or
                       r["nonpositive_page_numbers"] or r["out_of_order_page_numbers"]) for r in records),
                   pdf_mapping_counts=dict(Counter(r["pdf_mapping"] for r in records)),
                   script_classification_counts=dict(Counter(r["script_classification"] for r in records)),
                   exact_markdown_duplicate_groups=sum(count > 1 for count in digest_counts.values()),
                   pdfs_not_uniquely_matched_to_markdown=len(set(pdfs)-matched))
    return dict(schema="personal-kb-corpus-audit-v1", summary=summary, documents=records,
                pdfs_not_uniquely_matched=[p.relative_to(pdf_root).as_posix() for p in pdfs if p not in matched],
                safety=dict(api_calls=0, credentials_read=False, source_documents_modified=False,
                            customer_deliverables_modified=False, raw_text_copied=False),
                interpretation=["Counts and script ratios are not OCR/semantic correctness scores.",
                                "Source headers/page markers do not independently certify PDF printed pagination.",
                                "PDFs not matched may be duplicates or naming variants, not necessarily failed extraction.",
                                "Identical Markdown hash detects exact duplicates only; no semantic deduplication.",
                                "This inventory is not an index, retrieval evaluation, or working RAG app."])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown-dir", type=Path, required=True)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Inventory output already exists; choose a new snapshot path")
    report = inventory(args.markdown_dir, args.pdf_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(event="corpus_audit_complete", output=str(args.output.resolve()), **report["summary"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
