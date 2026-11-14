"""Heading-aware chunking (FR-02). OWNER: Member 2.

Strategy (recommended): never cross a section boundary; inside a section split on
paragraphs/sentences up to ``max_tokens`` with ``overlap_tokens`` overlap. Prefix
nothing into ``Chunk.text`` - the section path is kept in ``Chunk.section`` and
retrieval may choose to index ``section + text``.
"""

from __future__ import annotations

from src.schemas import Chunk, ParsedDocument


def count_tokens(text: str) -> int:
    """Cheap approximation (whitespace words * 1.3) is fine; document the choice."""
    raise NotImplementedError("Member 2: implement count_tokens")


def split_section_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split one section's text into windows on sentence boundaries."""
    raise NotImplementedError("Member 2: implement split_section_text")


def chunk_document(doc: ParsedDocument, max_tokens: int = 300, overlap_tokens: int = 50,
                   min_chunk_chars: int = 40) -> list[Chunk]:
    """Produce Chunks with ids '{doc_id}::{ordinal:04d}', inheriting doc metadata
    (title, source, version, effective_date) and section/page."""
    raise NotImplementedError("Member 2: implement chunk_document")


def chunk_documents(docs: list[ParsedDocument], max_tokens: int = 300, overlap_tokens: int = 50,
                    min_chunk_chars: int = 40) -> list[Chunk]:
    """Chunk many documents; chunk_ids must be globally unique."""
    raise NotImplementedError("Member 2: implement chunk_documents")
