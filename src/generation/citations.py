"""Citation building and validation (FR-05). OWNER: Member 4."""

from __future__ import annotations

from src.schemas import Citation, RetrievedPassage


def select_excerpt(passage_text: str, claim_text: str, max_chars: int = 240) -> str:
    """Pick the sentence(s) of the passage with the highest token overlap with the claim.
    The excerpt MUST be a verbatim substring of ``passage_text``."""
    raise NotImplementedError("Member 4: implement select_excerpt")


def build_citations(passage_numbers_by_claim: list[tuple[str, list[int]]],
                    passages: list[RetrievedPassage]) -> list[Citation]:
    """Map the LLM's passage numbers (1-based index into ``passages``) to Citation
    objects. Drop numbers that are out of range (hallucinated citations) and log them.
    Renumber citations 1..n in order of first appearance."""
    raise NotImplementedError("Member 4: implement build_citations")
