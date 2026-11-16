"""Member 4 - generation, citation and verification tests.

Never call the real LLM in unit tests: use ExtractiveClient or a fake client.
Remove the xfail marker once a function is implemented.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.errors import ConfigError, LLMError, LLMOutputParseError
from src.generation import citations, generator, prompts, verification
from src.generation.llm_client import (
    AnthropicClient,
    ExtractiveClient,
    GeminiClient,
    get_llm_client,
    parse_user_prompt,
)
from src.generation.prompts import LLMAnswer, LLMClaim
from src.schemas import Claim


class FakeClient:
    """Returns a fixed LLMAnswer (or raises) so generator logic is tested without a network."""

    model_name = "fake-llm"

    def __init__(self, answer: LLMAnswer | None = None, error: Exception | None = None) -> None:
        self.answer, self.error, self.calls = answer, error, 0

    def generate_structured(self, system, user, schema):
        self.calls += 1
        if self.error:
            raise self.error
        return self.answer


def test_prompt_numbers_passages(passages):
    text = prompts.build_user_prompt("rate limit?", passages)
    assert "[1]" in text and "[2]" in text and "rate limit?" in text


def test_prompt_includes_metadata_and_version(passages):
    text = prompts.build_user_prompt("rate limit?", passages, version="2.0")
    assert "API Usage Limits (v2.0) > API Usage Limits > Rate limits, v2.0" in text
    assert "product version 2.0" in text


def test_prompt_round_trips_for_extractive_client(passages):
    question, parsed = parse_user_prompt(prompts.build_user_prompt("rate limit?", passages))
    assert question == "rate limit?"
    assert [n for n, _ in parsed] == [1, 2, 3]
    assert parsed[0][1] == passages[0].chunk.text


def test_out_of_range_citations_are_dropped(passages):
    cits = citations.build_citations([("claim", [1, 99])], passages)
    assert [c.chunk_id for c in cits] == [passages[0].chunk.chunk_id]


def test_citations_renumbered_in_order_of_first_use(passages):
    cits = citations.build_citations([("a", [2]), ("b", [1, 2])], passages)
    assert [(c.citation_id, c.chunk_id) for c in cits] == [
        (1, passages[1].chunk.chunk_id), (2, passages[0].chunk.chunk_id)]


def test_excerpt_is_verbatim(passages):
    text = passages[0].chunk.text
    assert citations.select_excerpt(text, "300 requests per minute") in text


def test_excerpt_prefers_matching_sentence(passages):
    excerpt = citations.select_excerpt(passages[2].chunk.text, "the reset link is valid for 30 minutes")
    assert "30 minutes" in excerpt and excerpt in passages[2].chunk.text


def test_number_not_in_passage_is_unsupported(passages):
    claim = Claim(text="The limit is 999 requests per minute.", citation_ids=[1])
    assert verification.claim_support_score(claim, passages[:1]) < 0.5


def test_supported_claim_scores_high(passages):
    claim = Claim(text="Standard plan accounts can send up to 300 requests per minute (v2.0).")
    assert verification.claim_support_score(claim, passages[:1]) >= 0.6


def test_groundedness_score():
    claims = [Claim(text="a", supported=True), Claim(text="b", supported=False)]
    assert verification.groundedness_score(claims) == 0.5


def test_generate_with_no_passages_is_insufficient(settings):
    g = generator.AnswerGenerator(settings.generation, client=ExtractiveClient())
    assert g.generate("anything?", []).insufficient_evidence is True


def test_extractive_answer_is_cited_and_grounded(settings, passages):
    g = generator.AnswerGenerator(settings.generation, client=ExtractiveClient())
    ans = g.generate("How many API requests can I make per minute?", passages)
    assert not ans.insufficient_evidence
    assert ans.citations and all(c.excerpt in next(p.chunk.text for p in passages
                                                   if p.chunk.chunk_id == c.chunk_id) for c in ans.citations)
    assert ans.groundedness_score == 1.0
    assert "[1]" in ans.answer_text


def test_extractive_unrelated_question_is_insufficient(settings, passages):
    g = generator.AnswerGenerator(settings.generation, client=ExtractiveClient())
    assert g.generate("Does the product support Kubernetes deployment?", passages).insufficient_evidence


def test_generate_renumbers_and_drops_hallucinated_citation(settings, passages):
    llm = LLMAnswer(
        answer="The v1.x FAQ allows 100 requests per minute [2] [7].",
        claims=[LLMClaim(text="The v1.x FAQ allows 100 requests per minute.", passage_numbers=[2, 7])],
        insufficient_evidence=False, conflict=False,
    )
    ans = generator.AnswerGenerator(settings.generation, client=FakeClient(llm)).generate("limit?", passages)
    assert ans.answer_text == "The v1.x FAQ allows 100 requests per minute [1]."
    assert [c.chunk_id for c in ans.citations] == [passages[1].chunk.chunk_id]
    assert ans.citations[0].citation_id == 1 and ans.model == "fake-llm"


def test_generate_removes_unsupported_claim(settings, passages):
    llm = LLMAnswer(
        answer="Standard plan accounts can send 300 requests per minute [1]. Enterprise gets 5000 [1].",
        claims=[LLMClaim(text="Standard plan accounts can send 300 requests per minute.", passage_numbers=[1]),
                LLMClaim(text="Enterprise accounts get 5000 requests per minute.", passage_numbers=[1])],
        insufficient_evidence=False, conflict=False,
    )
    ans = generator.AnswerGenerator(settings.generation, client=FakeClient(llm)).generate("limit?", passages)
    assert "5000" not in ans.answer_text and "300" in ans.answer_text
    assert ans.groundedness_score == 0.5
    assert [c.supported for c in ans.claims] == [True, False]


def test_generate_all_unsupported_is_insufficient(settings, passages):
    llm = LLMAnswer(answer="It is 999 [1].", claims=[LLMClaim(text="It is 999.", passage_numbers=[1])],
                    insufficient_evidence=False, conflict=True)
    ans = generator.AnswerGenerator(settings.generation, client=FakeClient(llm)).generate("limit?", passages)
    assert ans.insufficient_evidence and ans.citations == [] and ans.llm_reported_conflict


def test_llm_error_falls_back_to_extractive(settings, passages):
    fake = FakeClient(error=LLMError("boom"))
    ans = generator.AnswerGenerator(settings.generation, client=fake).generate(
        "How many API requests can I make per minute?", passages)
    assert fake.calls == 1 and ans.model == "extractive-fallback" and ans.citations


ALL_KEY_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY", "SARVAM_API_KEY")


def _clear_keys(monkeypatch):
    for var in ALL_KEY_VARS:
        monkeypatch.delenv(var, raising=False)


def test_get_llm_client_without_key_uses_extractive(settings, monkeypatch):
    _clear_keys(monkeypatch)
    assert isinstance(get_llm_client(settings.generation), ExtractiveClient)


def test_get_llm_client_anthropic_without_key_but_gemini_key(settings, monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test-not-a-real-key")
    assert isinstance(get_llm_client(settings.generation), GeminiClient)


def test_get_llm_client_with_key_uses_anthropic(settings, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-real-key")
    assert isinstance(get_llm_client(settings.generation), AnthropicClient)


def test_get_llm_client_gemini_with_key(settings, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-not-a-real-key")
    settings.generation.provider = "gemini"
    client = get_llm_client(settings.generation)
    assert isinstance(client, GeminiClient) and client.model_name == "gemini-3.5-flash-lite"


def test_get_llm_client_gemini_without_key_uses_extractive(settings, monkeypatch):
    _clear_keys(monkeypatch)
    settings.generation.provider = "gemini"
    assert isinstance(get_llm_client(settings.generation), ExtractiveClient)


class _FakeGeminiModels:
    def __init__(self, response=None, error=None):
        self.response, self.error = response, error

    def generate_content(self, *, model, contents, config):
        if self.error:
            raise self.error
        return self.response


def _gemini_with(settings, response=None, error=None) -> GeminiClient:
    client = GeminiClient(settings.generation)
    client._client = SimpleNamespace(models=_FakeGeminiModels(response, error))
    return client


def _gemini_response(finish: str, text: str = "", parsed=None):
    return SimpleNamespace(candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))],
                           parsed=parsed, text=text)


def test_gemini_parses_json_text(settings):
    payload = LLMAnswer(answer="a [1]", claims=[LLMClaim(text="a", passage_numbers=[1])],
                        insufficient_evidence=False, conflict=False)
    client = _gemini_with(settings, _gemini_response("STOP", text=payload.model_dump_json()))
    assert client.generate_structured("sys", "user", LLMAnswer) == payload


def test_gemini_safety_block_is_llm_error(settings):
    with pytest.raises(LLMError):
        _gemini_with(settings, _gemini_response("SAFETY")).generate_structured("s", "u", LLMAnswer)


def test_gemini_truncated_or_bad_json_is_parse_error(settings):
    with pytest.raises(LLMOutputParseError):
        _gemini_with(settings, _gemini_response("MAX_TOKENS")).generate_structured("s", "u", LLMAnswer)
    with pytest.raises(LLMOutputParseError):
        _gemini_with(settings, _gemini_response("STOP", text="{not json")).generate_structured("s", "u", LLMAnswer)


def test_gemini_api_error_becomes_llm_error(settings):
    from google.genai import errors
    err = errors.ClientError(429, {"error": {"message": "quota"}})
    with pytest.raises(LLMError):
        _gemini_with(settings, error=err).generate_structured("s", "u", LLMAnswer)


def test_get_llm_client_unknown_provider(settings):
    settings.generation.provider = "nope"
    with pytest.raises(ConfigError):
        get_llm_client(settings.generation)


def test_markers_inside_claim_text_are_not_numbers(settings, passages):
    llm = LLMAnswer(
        answer="Standard plan accounts can send 300 requests per minute [1]. The v1.x FAQ says 100 [2].",
        claims=[LLMClaim(text="Standard plan accounts can send 300 requests per minute [1].", passage_numbers=[1]),
                LLMClaim(text="The v1.x FAQ allows 100 requests per minute [2].", passage_numbers=[2])],
        insufficient_evidence=False, conflict=True,
    )
    ans = generator.AnswerGenerator(settings.generation, client=FakeClient(llm)).generate("limit?", passages)
    assert ans.groundedness_score == 1.0 and not ans.insufficient_evidence
    assert all("[" not in c.text for c in ans.claims)
