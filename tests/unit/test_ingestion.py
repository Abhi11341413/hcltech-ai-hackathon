"""Member 2 - ingestion & chunking tests.

Tests are marked xfail(NotImplementedError) until implemented. When you implement a
function, REMOVE its marker so the test must genuinely pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.errors import UnsupportedFormatError
from src.ingestion import chunking, cleaning, loaders
from src.schemas import FileType

todo = pytest.mark.xfail(raises=NotImplementedError, reason="Member 2: not implemented yet", strict=False)


@todo
def test_detect_file_type():
    assert loaders.detect_file_type(Path("a.MD")) == FileType.MARKDOWN
    with pytest.raises(UnsupportedFormatError):
        loaders.detect_file_type(Path("a.docx"))


@todo
def test_markdown_preserves_heading_hierarchy(sample_docs_dir):
    doc = loaders.load_document(sample_docs_dir / "api_usage_limits_v2.md")
    assert doc.file_type == FileType.MARKDOWN
    assert any(s.heading_path[-1:] == ["Rate limits"] for s in doc.sections)
    assert doc.version == "2.0"


@todo
def test_html_strips_scripts_and_keeps_sections(sample_docs_dir):
    doc = loaders.load_document(sample_docs_dir / "account_access.html")
    text = " ".join(s.text for s in doc.sections)
    assert "<script" not in text and "Password reset" in " ".join(" ".join(s.heading_path) for s in doc.sections)


@todo
def test_load_directory_reports_failures_without_crashing(tmp_path):
    (tmp_path / "ok.txt").write_text("GETTING STARTED\nHello world, this is a valid document.")
    (tmp_path / "bad.docx").write_bytes(b"not supported")
    docs, failures = loaders.load_directory(tmp_path)
    assert len(docs) == 1 and failures[0].error_code == "UNSUPPORTED_FORMAT"


@todo
def test_clean_text_collapses_whitespace():
    assert cleaning.clean_text("Rate   limit\n\n\nis  300") == "Rate limit\n\nis 300"


@todo
def test_chunk_ids_unique_and_metadata_inherited():
    from src import mocks
    chunks = chunking.chunk_document(mocks.mock_parsed_document(), max_tokens=20, overlap_tokens=5)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.version == "2.0" and c.doc_title for c in chunks)
