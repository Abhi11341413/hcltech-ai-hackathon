"""Heading-aware chunking (FR-02). OWNER: Member 2.

Strategy: never cross a section boundary; inside a section split on sentence / line
boundaries up to ``max_tokens`` with ``overlap_tokens`` overlap. Nothing is prefixed into
``Chunk.text`` (so citation excerpts stay verbatim substrings); the heading path is kept in
``Chunk.section`` and the retrievers index ``doc_title + section + text`` ("contextual
header injection" at index time, not in the stored passage).
"""

from __future__ import annotations

import math
import re

from src.schemas import Chunk, ParsedDocument

# Sentence end (. ! ?) followed by whitespace, or any line break. Line breaks matter for
# TXT FAQs ("Q: ..."/"A: ...") and lists, where lines are the natural units.
_UNIT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+")


def count_tokens(text: str) -> int:
    """Approximate token count: whitespace words * 1.3, rounded up.

    Good enough to size chunks for a ~256-token embedding window without loading a
    tokenizer; English text averages ~1.3 sub-word tokens per word.
    """
    return math.ceil(len(text.split()) * 1.3)


def _units(text: str) -> list[str]:
    """Split text into sentences/lines; split any unit longer than the window by words later."""
    return [u.strip() for u in _UNIT_SPLIT.split(text) if u and u.strip()]


def _split_long_unit(unit: str, max_tokens: int) -> list[str]:
    words = unit.split()
    step = max(1, int(max_tokens / 1.3))
    return [" ".join(words[i:i + step]) for i in range(0, len(words), step)]


def split_section_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split one section's text into windows on sentence boundaries.

    Windows are packed greedily up to ``max_tokens``. Each new window starts with the last
    sentences of the previous one, up to ``overlap_tokens``, so a fact split across a
    boundary still appears whole in one window. Text that fits in one window is returned
    unchanged (as a single element) so it stays a verbatim copy of the section.

    Raises:
        ValueError: max_tokens < 1 or overlap_tokens < 0 or overlap_tokens >= max_tokens.
    """
    if max_tokens < 1 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("require max_tokens >= 1 and 0 <= overlap_tokens < max_tokens")
    text = text.strip()
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [text]

    units: list[str] = []
    for unit in _units(text):
        units.extend(_split_long_unit(unit, max_tokens) if count_tokens(unit) > max_tokens else [unit])

    windows: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and count_tokens(" ".join(current + [unit])) > max_tokens:
            windows.append(" ".join(current))
            overlap: list[str] = []
            for prev in reversed(current):
                if count_tokens(" ".join([prev] + overlap)) > overlap_tokens:
                    break
                overlap.insert(0, prev)
            # keep the overlap only if the new unit still fits next to it
            current = overlap if count_tokens(" ".join(overlap + [unit])) <= max_tokens else []
        current.append(unit)
    if current:
        windows.append(" ".join(current))
    return windows


def chunk_document(doc: ParsedDocument, max_tokens: int = 300, overlap_tokens: int = 50,
                   min_chunk_chars: int = 40) -> list[Chunk]:
    """Produce Chunks with ids '{doc_id}::{ordinal:04d}', inheriting doc metadata
    (title, source, version, effective_date) and section/page.

    Windows shorter than ``min_chunk_chars`` are dropped, except when they are the only
    content of their section (a short section such as "Sessions expire after 8 hours"
    is still a real answer).
    """
    chunks: list[Chunk] = []
    for section in doc.sections:
        windows = split_section_text(section.text, max_tokens, overlap_tokens)
        keep = [w for w in windows if len(w) >= min_chunk_chars] or windows[:1]
        for window in keep:
            ordinal = len(chunks)
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}::{ordinal:04d}",
                doc_id=doc.doc_id,
                doc_title=doc.title,
                source=doc.source,
                section=" > ".join(section.heading_path),
                page=section.page,
                text=window,
                version=doc.version,
                effective_date=doc.effective_date,
                ordinal=ordinal,
            ))
    return chunks


def chunk_documents(docs: list[ParsedDocument], max_tokens: int = 300, overlap_tokens: int = 50,
                    min_chunk_chars: int = 40) -> list[Chunk]:
    """Chunk many documents; chunk_ids are globally unique because doc_ids are
    (``loaders.load_directory`` rejects duplicate doc_ids).

    Raises:
        ValueError: two documents share a doc_id.
    """
    seen: set[str] = set()
    chunks: list[Chunk] = []
    for doc in docs:
        if doc.doc_id in seen:
            raise ValueError(f"duplicate doc_id '{doc.doc_id}'")
        seen.add(doc.doc_id)
        chunks.extend(chunk_document(doc, max_tokens, overlap_tokens, min_chunk_chars))
    return chunks
