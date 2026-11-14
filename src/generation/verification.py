"""Groundedness verification: is every claim supported by its cited passages? OWNER: Member 4.

Recommended (deterministic, explainable): for each claim compute
  support = max over cited passages of  token-recall(claim content words, passage)
optionally combined with embedding cosine (reuse Member 3's Embedder).
Numbers in the claim (e.g. "300") MUST appear in a cited passage, else unsupported.

We use the token-recall version only (no embeddings), so every score can be explained
by listing the claim words that were / were not found in the passage.
"""

from __future__ import annotations

import re

from src.reasoning.text_utils import content_terms
from src.schemas import Citation, Claim, RetrievedPassage

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)*")
_MARKER_RE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")


def strip_citation_markers(text: str) -> str:
    """'300 requests per minute [1].' -> '300 requests per minute.' (markers are not claim content)."""
    return re.sub(r"\s+([.,;:!?])", r"\1", _MARKER_RE.sub("", text)).strip()


def _passage_haystack(passage: RetrievedPassage) -> str:
    """Passage text plus its metadata, so a claim such as '(v2.0)' can be supported by the version."""
    chunk = passage.chunk
    meta = [chunk.doc_title, chunk.section, chunk.version or "",
            chunk.effective_date.isoformat() if chunk.effective_date else ""]
    return " ".join([chunk.text, *meta])


def claim_support_score(claim: Claim, cited_passages: list[RetrievedPassage]) -> float:
    """Return support in [0, 1].

    For each cited passage: 0 if any number in the claim is missing from it, otherwise
    the fraction of the claim's content terms found in it. The claim's score is the best
    passage score. No cited passages -> 0.
    """
    if not cited_passages:
        return 0.0
    text = strip_citation_markers(claim.text)  # '[1]' must not count as the number 1
    claim_terms = content_terms(text)
    claim_numbers = set(_NUMBER_RE.findall(text))
    best = 0.0
    for passage in cited_passages:
        haystack = _passage_haystack(passage)
        if claim_numbers - set(_NUMBER_RE.findall(haystack)):
            continue  # a number the passage never states -> this passage does not support the claim
        if not claim_terms:
            return 1.0
        recall = len(claim_terms & content_terms(haystack)) / len(claim_terms)
        best = max(best, recall)
    return round(min(max(best, 0.0), 1.0), 3)


def verify_claims(claims: list[Claim], citations: list[Citation],
                  passages: list[RetrievedPassage], threshold: float = 0.6) -> list[Claim]:
    """Return claims with ``supported`` and ``support_score`` filled in.

    A claim's citation ids are resolved through ``citations`` to chunk ids, and from there
    to the passages; ids that resolve to nothing are ignored (so they cannot add support).
    """
    by_chunk = {p.chunk.chunk_id: p for p in passages}
    chunk_of = {c.citation_id: c.chunk_id for c in citations}
    verified: list[Claim] = []
    for claim in claims:
        cited = [by_chunk[chunk_of[i]] for i in claim.citation_ids
                 if i in chunk_of and chunk_of[i] in by_chunk]
        score = claim_support_score(claim, cited)
        verified.append(claim.model_copy(update={"supported": score >= threshold, "support_score": score}))
    return verified


def groundedness_score(claims: list[Claim]) -> float:
    """Fraction of claims marked supported (0.0 if no claims)."""
    if not claims:
        return 0.0
    return sum(1 for c in claims if c.supported) / len(claims)
