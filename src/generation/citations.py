"""Citation building and validation (FR-05). OWNER: Member 4."""

from __future__ import annotations

import re

from src.config import get_logger
from src.reasoning.text_utils import content_terms, split_sentences
from src.schemas import Citation, RetrievedPassage

logger = get_logger(__name__)

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)*")


def _overlap(sentence: str, claim_terms: set[str], claim_numbers: set[str]) -> float:
    """Shared content terms, with a shared number counting double (numbers are what users check)."""
    terms = content_terms(sentence)
    numbers = set(_NUMBER_RE.findall(sentence))
    return len(terms & claim_terms) + 2 * len(numbers & claim_numbers)


def _truncate(text: str, max_chars: int) -> str:
    """Cut at a word boundary; the result is still a prefix (so a substring) of ``text``."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    space = cut.rfind(" ")
    return cut[:space] if space > max_chars // 2 else cut


def select_excerpt(passage_text: str, claim_text: str, max_chars: int = 240) -> str:
    """Pick the sentence(s) of the passage with the highest token overlap with the claim.
    The excerpt MUST be a verbatim substring of ``passage_text``.

    Input: the chunk text and the claim it should support. Output: the best sentence,
    extended with the following sentence when both fit in ``max_chars``.
    """
    sentences = split_sentences(passage_text)
    if not sentences:
        return _truncate(passage_text.strip(), max_chars)
    claim_terms = content_terms(claim_text)
    claim_numbers = set(_NUMBER_RE.findall(claim_text))
    best = max(range(len(sentences)), key=lambda i: _overlap(sentences[i], claim_terms, claim_numbers))

    excerpt = sentences[best]
    if best + 1 < len(sentences):
        # Extend to the next sentence only if the two are adjacent verbatim in the passage.
        start = passage_text.find(excerpt)
        end = passage_text.find(sentences[best + 1], start + len(excerpt))
        if start >= 0 and end >= 0:
            combined = passage_text[start:end + len(sentences[best + 1])]
            if len(combined) <= max_chars and _overlap(sentences[best + 1], claim_terms, claim_numbers) > 0:
                excerpt = combined
    return _truncate(excerpt, max_chars)


def build_citations(passage_numbers_by_claim: list[tuple[str, list[int]]],
                    passages: list[RetrievedPassage]) -> list[Citation]:
    """Map the LLM's passage numbers (1-based index into ``passages``) to Citation
    objects. Drop numbers that are out of range (hallucinated citations) and log them.
    Renumber citations 1..n in order of first appearance.

    The excerpt of each citation is chosen for the first claim that cites that passage.
    """
    citations: list[Citation] = []
    seen: set[int] = set()
    for claim_text, numbers in passage_numbers_by_claim:
        for n in numbers:
            if not 1 <= n <= len(passages):
                logger.warning("dropping out-of-range citation [%d] (only %d passages)", n, len(passages))
                continue
            if n in seen:
                continue
            seen.add(n)
            chunk = passages[n - 1].chunk
            citations.append(Citation(
                citation_id=len(citations) + 1,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                section=chunk.section,
                page=chunk.page,
                source=chunk.source,
                excerpt=select_excerpt(chunk.text, claim_text),
            ))
    return citations
