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


def test_detect_file_type():
    assert loaders.detect_file_type(Path("a.MD")) == FileType.MARKDOWN
    with pytest.raises(UnsupportedFormatError):
        loaders.detect_file_type(Path("a.docx"))


def test_markdown_preserves_heading_hierarchy(sample_docs_dir):
    doc = loaders.load_document(sample_docs_dir / "api_usage_limits_v2.md")
    assert doc.file_type == FileType.MARKDOWN
    assert any(s.heading_path[-1:] == ["Rate limits"] for s in doc.sections)
    assert doc.version == "2.0"


def test_html_strips_scripts_and_keeps_sections(sample_docs_dir):
    doc = loaders.load_document(sample_docs_dir / "account_access.html")
    text = " ".join(s.text for s in doc.sections)
    assert "<script" not in text and "Password reset" in " ".join(" ".join(s.heading_path) for s in doc.sections)


def test_load_directory_reports_failures_without_crashing(tmp_path):
    (tmp_path / "ok.txt").write_text("GETTING STARTED\nHello world, this is a valid document.")
    (tmp_path / "bad.docx").write_bytes(b"not supported")
    docs, failures = loaders.load_directory(tmp_path)
    assert len(docs) == 1 and failures[0].error_code == "UNSUPPORTED_FORMAT"


def test_clean_text_collapses_whitespace():
    assert cleaning.clean_text("Rate   limit\n\n\nis  300") == "Rate limit\n\nis 300"


def test_chunk_ids_unique_and_metadata_inherited():
    from src import mocks
    chunks = chunking.chunk_document(mocks.mock_parsed_document(), max_tokens=20, overlap_tokens=5)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.version == "2.0" and c.doc_title for c in chunks)


def test_pdf_sections_keep_page_numbers(sample_docs_dir):
    doc = loaders.load_document(sample_docs_dir / "sso_admin_guide_v3.pdf")
    assert doc.file_type == FileType.PDF and doc.version == "3.0"
    pages = {s.heading_path[-1]: s.page for s in doc.sections if len(s.heading_path) == 2}
    assert pages["1. Overview"] == 1 and pages["5. Session policy"] == 2


def test_txt_caps_headings_match_eval_labels(sample_docs_dir):
    doc = loaders.load_document(sample_docs_dir / "support_faq_v1.txt")
    assert doc.version == "1.x" and str(doc.effective_date) == "2025-03-15"
    assert ["LIMITS"] in [s.heading_path for s in doc.sections]


def test_extract_effective_date_ignores_unrelated_dates():
    text = "Released 1 October 2026. The v1 API will be switched off on 31 March 2027."
    assert str(cleaning.extract_effective_date(text)) == "2026-10-01"
    assert cleaning.extract_version("Policy", "Policy version 2026.1. Effective") == "2026.1"


def test_split_section_text_overlaps_and_respects_limit():
    text = " ".join(f"Sentence number {i} has five words." for i in range(30))
    windows = chunking.split_section_text(text, max_tokens=40, overlap_tokens=10)
    assert len(windows) > 1
    assert all(chunking.count_tokens(w) <= 40 for w in windows)
    assert windows[0].split(". ")[-1].rstrip(".") in windows[1]  # last sentence carried over


def test_whole_corpus_chunks_have_unique_ids(sample_docs_dir):
    docs, failures = loaders.load_directory(sample_docs_dir)
    chunks = chunking.chunk_documents(docs)
    assert not failures and len(docs) == 13
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all("must be stripped" not in c.text for c in chunks)
