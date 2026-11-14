"""Groundedness verification: is every claim supported by its cited passages? OWNER: Member 4.

Recommended (deterministic, explainable): for each claim compute
  support = max over cited passages of  token-recall(claim content words, passage)
optionally combined with embedding cosine (reuse Member 3's Embedder).
Numbers in the claim (e.g. "300") MUST appear in a cited passage, else unsupported.
"""

from __future__ import annotations

from src.schemas import Citation, Claim, RetrievedPassage


def claim_support_score(claim: Claim, cited_passages: list[RetrievedPassage]) -> float:
    """Return support in [0, 1]."""
    raise NotImplementedError("Member 4: implement claim_support_score")


def verify_claims(claims: list[Claim], citations: list[Citation],
                  passages: list[RetrievedPassage], threshold: float = 0.6) -> list[Claim]:
    """Return claims with ``supported`` and ``support_score`` filled in."""
    raise NotImplementedError("Member 4: implement verify_claims")


def groundedness_score(claims: list[Claim]) -> float:
    """Fraction of claims marked supported (0.0 if no claims)."""
    raise NotImplementedError("Member 4: implement groundedness_score")
