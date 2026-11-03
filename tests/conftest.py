from __future__ import annotations

import pytest

from literature_kb.config import Settings
from literature_kb.store import Store


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "sources"
    folder.mkdir()
    return folder


@pytest.fixture
def settings(source, tmp_path):
    return Settings(source, tmp_path / "index", chunk_chars=180, overlap_chars=30)


@pytest.fixture
def store(settings):
    return Store(settings)


@pytest.fixture
def write_source(source):
    def write(name="test.md", body=None):
        text = body or (
            "# Test\r\n> Source: `test.pdf`\r\n## Page 1\r\n"
            + "证据定位需要保留原始文本字符位置，文档替换后旧引用应该失效。" * 10
            + "\r\n## Page 2\r\n"
            + "Версия документа и фрагменты сохраняются. Мономаха и Мономах. " * 8
        )
        (source / name).write_bytes(text.encode("utf-8"))
        return text

    return write
