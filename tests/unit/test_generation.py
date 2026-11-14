"""Member 4 - generation, citation and verification tests.

Never call the real LLM in unit tests: use ExtractiveClient or a fake client.
Remove the xfail marker once a function is implemented.
"""

from __future__ import annotations

import pytest

from src.generation import citations, generator, prompts, verification
from src.generation.llm_client import ExtractiveClient
from src.schemas import Claim

todo = pytest.mark.xfail(raises=NotImplementedError, reason="Member 4: not implemented yet", strict=False)


@todo
def test_prompt_numbers_passages(passages):
    text = prompts.build_user_prompt("rate limit?", passages)
    assert "[1]" in text and "[2]" in text and "rate limit?" in text


@todo
def test_out_of_range_citations_are_dropped(passages):
    cits = citations.build_citations([("claim", [1, 99])], passages)
    assert [c.chunk_id for c in cits] == [passages[0].chunk.chunk_id]


@todo
def test_excerpt_is_verbatim(passages):
    text = passages[0].chunk.text
    assert citations.select_excerpt(text, "300 requests per minute") in text


@todo
def test_number_not_in_passage_is_unsupported(passages):
    claim = Claim(text="The limit is 999 requests per minute.", citation_ids=[1])
    assert verification.claim_support_score(claim, passages[:1]) < 0.5


@todo
def test_groundedness_score():
    claims = [Claim(text="a", supported=True), Claim(text="b", supported=False)]
    assert verification.groundedness_score(claims) == 0.5


@todo
def test_generate_with_no_passages_is_insufficient(settings):
    g = generator.AnswerGenerator(settings.generation, client=ExtractiveClient())
    assert g.generate("anything?", []).insufficient_evidence is True
