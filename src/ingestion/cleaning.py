"""Text cleaning and metadata extraction (FR-02, supports FR-06). OWNER: Member 2."""

from __future__ import annotations

from datetime import date

from src.schemas import ParsedDocument


def clean_text(text: str) -> str:
    """Normalise unicode (NFKC), fix hyphenated line breaks, collapse whitespace,
    remove repeated headers/footers and page numbers. Must NOT drop headings."""
    raise NotImplementedError("Member 2: implement clean_text")


def extract_version(title: str, text: str) -> str | None:
    """Detect a product/doc version, e.g. 'v2.0', 'Version 3.1', 'Release 2024.2' -> '2.0'."""
    raise NotImplementedError("Member 2: implement extract_version")


def extract_effective_date(text: str) -> date | None:
    """Detect 'Effective from 1 July 2026', 'Last updated: 2026-07-01', etc."""
    raise NotImplementedError("Member 2: implement extract_effective_date")


def clean_document(doc: ParsedDocument) -> ParsedDocument:
    """Return a new ParsedDocument with cleaned sections, version and effective_date filled.
    Empty sections are removed. Raise EmptyDocumentError if nothing remains."""
    raise NotImplementedError("Member 2: implement clean_document")
