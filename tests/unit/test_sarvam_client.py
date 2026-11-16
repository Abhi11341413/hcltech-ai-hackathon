"""SarvamClient tests. The HTTP layer is faked, so no key or network is needed.

A live smoke test runs only when RUN_LLM_TESTS=1 and SARVAM_API_KEY are set.
"""

from __future__ import annotations

import io
import json
import os
import urllib.error

import pytest

from src.config import GenerationSettings
from src.errors import LLMError, LLMOutputParseError
from src.generation import sarvam_client
from src.generation.llm_client import ExtractiveClient, get_llm_client
from src.generation.prompts import LLMAnswer
from src.generation.sarvam_client import SarvamClient, extract_json_text, strict_json_schema

GOOD = {"answer": "Standard plan: 300 requests per minute [1].",
        "claims": [{"text": "Standard plan: 300 requests per minute.", "passage_numbers": [1]}],
        "insufficient_evidence": False, "conflict": False}


def ok_payload(content: str, finish: str = "stop") -> dict:
    return {"choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": content, "reasoning_content": "thinking..."}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 40}}


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(status: int, code: str = "x", message: str = "m", retry_after: str | None = None):
    headers = {"Retry-After": retry_after} if retry_after else {}
    body = json.dumps({"error": {"code": code, "message": message}}).encode()
    return urllib.error.HTTPError("https://api.sarvam.ai", status, "err", headers, io.BytesIO(body))


@pytest.fixture
def fake_http(monkeypatch):
    """Queue responses (dicts) or exceptions; records every request sent."""
    calls: list[dict] = []
    queue: list = []

    def urlopen(request, timeout=None):
        calls.append({"url": request.full_url, "headers": dict(request.header_items()),
                      "body": json.loads(request.data.decode()), "timeout": timeout})
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeResponse(json.dumps(item).encode())

    monkeypatch.setattr(sarvam_client.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(sarvam_client.time, "sleep", lambda s: None)
    return calls, queue


def client(**overrides) -> SarvamClient:
    return SarvamClient(GenerationSettings(provider="sarvam", **overrides), api_key="test-not-a-real-key")


def test_request_matches_sarvam_api_contract(fake_http):
    calls, queue = fake_http
    queue.append(ok_payload(json.dumps(GOOD)))
    result = client().generate_structured("SYSTEM", "USER", LLMAnswer)
    assert result.answer.startswith("Standard plan") and result.claims[0].passage_numbers == [1]
    sent = calls[0]
    assert sent["url"] == "https://api.sarvam.ai/v1/chat/completions"
    assert sent["headers"]["Api-subscription-key"] == "test-not-a-real-key"
    body = sent["body"]
    assert body["model"] == "sarvam-105b" and body["max_tokens"] == 4096 and body["reasoning_effort"] == "low"
    assert body["messages"] == [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "USER"}]
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["name"] == "LLMAnswer"
    assert fmt["json_schema"]["strict"] is True


def test_model_setting_used_only_when_it_is_a_sarvam_model():
    assert client(model="claude-opus-5-5").model_name == "sarvam-105b"
    assert client(model="sarvam-105b-conversations").model_name == "sarvam-105b-conversations"


def test_reasoning_can_be_disabled(fake_http):
    calls, queue = fake_http
    queue.append(ok_payload(json.dumps(GOOD)))
    client(effort="none").generate_structured("s", "u", LLMAnswer)
    assert "reasoning_effort" in calls[0]["body"] and calls[0]["body"]["reasoning_effort"] is None


def test_strict_schema_is_closed_and_inlined():
    schema = strict_json_schema(LLMAnswer)
    assert "$defs" not in json.dumps(schema) and "$ref" not in json.dumps(schema)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"answer", "claims", "insufficient_evidence", "conflict"}
    claim = schema["properties"]["claims"]["items"]
    assert claim["additionalProperties"] is False and set(claim["required"]) == {"text", "passage_numbers"}


@pytest.mark.parametrize("wrapped", [
    "```json\n{body}\n```",
    "<think>let me reason</think>\n{body}",
    "Here is the JSON: {body}",
])
def test_extract_json_text_handles_wrappers(wrapped):
    body = json.dumps(GOOD)
    assert json.loads(extract_json_text(wrapped.replace("{body}", body))) == GOOD


def test_schema_rejected_falls_back_to_json_mode(fake_http):
    calls, queue = fake_http
    queue += [http_error(400, "invalid_request_error", "response_format not supported"),
              ok_payload("```json\n" + json.dumps(GOOD) + "\n```")]
    result = client().generate_structured("SYSTEM", "USER", LLMAnswer)
    assert result.conflict is False and len(calls) == 2
    assert calls[1]["body"]["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in calls[1]["body"]["messages"][0]["content"]


def test_rate_limit_is_retried_once(fake_http):
    calls, queue = fake_http
    queue += [http_error(429, "rate_limit_exceeded_error", retry_after="1"), ok_payload(json.dumps(GOOD))]
    assert client().generate_structured("s", "u", LLMAnswer).answer
    assert len(calls) == 2


def test_persistent_overload_raises_llm_error(fake_http):
    _, queue = fake_http
    queue += [http_error(503, "model_overloaded"), http_error(503, "model_overloaded")]
    with pytest.raises(LLMError, match="HTTP 503"):
        client().generate_structured("s", "u", LLMAnswer)


def test_bad_key_gives_clear_error_without_retry(fake_http):
    calls, queue = fake_http
    queue.append(http_error(403, "invalid_api_key_error", "Invalid API key"))
    with pytest.raises(LLMError, match="SARVAM_API_KEY") as info:
        client().generate_structured("s", "u", LLMAnswer)
    assert len(calls) == 1 and info.value.details["status"] == "403"
    assert "test-not-a-real-key" not in str(info.value)


def test_cut_off_or_empty_output_is_a_parse_error(fake_http):
    _, queue = fake_http
    queue += [ok_payload("", finish="length"), ok_payload("")]
    with pytest.raises(LLMOutputParseError, match="max_tokens"):
        client().generate_structured("s", "u", LLMAnswer)
    with pytest.raises(LLMOutputParseError, match="empty"):
        client().generate_structured("s", "u", LLMAnswer)


def test_invalid_json_is_a_parse_error(fake_http):
    _, queue = fake_http
    queue.append(ok_payload('{"answer": "missing fields"}'))
    with pytest.raises(LLMOutputParseError):
        client().generate_structured("s", "u", LLMAnswer)


def test_content_filter_is_an_llm_error(fake_http):
    _, queue = fake_http
    queue.append(ok_payload("", finish="content_filter"))
    with pytest.raises(LLMError, match="declined"):
        client().generate_structured("s", "u", LLMAnswer)


def test_network_failure_is_an_llm_error(fake_http):
    _, queue = fake_http
    queue.append(urllib.error.URLError("dns failure"))
    with pytest.raises(LLMError, match="could not reach"):
        client().generate_structured("s", "u", LLMAnswer)


def test_missing_key_is_an_llm_error(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    with pytest.raises(LLMError, match="SARVAM_API_KEY"):
        SarvamClient(GenerationSettings(provider="sarvam")).generate_structured("s", "u", LLMAnswer)


# ------------------------------------------------------------------ provider selection
KEY_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY", "SARVAM_API_KEY")


@pytest.fixture
def no_keys(monkeypatch):
    for var in KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_provider_sarvam_with_key(no_keys):
    no_keys.setenv("SARVAM_API_KEY", "test-not-a-real-key")
    assert isinstance(get_llm_client(GenerationSettings(provider="sarvam")), SarvamClient)


def test_provider_sarvam_without_key_is_offline(no_keys):
    assert isinstance(get_llm_client(GenerationSettings(provider="sarvam")), ExtractiveClient)


def test_default_provider_falls_back_to_sarvam_when_it_is_the_only_key(no_keys):
    no_keys.setenv("SARVAM_API_KEY", "test-not-a-real-key")
    assert isinstance(get_llm_client(GenerationSettings()), SarvamClient)


# ------------------------------------------------------------------ live (opt-in)
@pytest.mark.llm
@pytest.mark.skipif(os.environ.get("RUN_LLM_TESTS") != "1" or not os.environ.get("SARVAM_API_KEY"),
                    reason="set RUN_LLM_TESTS=1 and SARVAM_API_KEY to call the real Sarvam API")
def test_live_sarvam_returns_valid_answer():
    from src.generation.prompts import SYSTEM_PROMPT
    user = ("<passages>\n[1] (Rate limits)\nStandard plan accounts may send up to 300 requests per minute.\n"
            "</passages>\n<question>How many requests per minute on Standard?</question>")
    result = SarvamClient(GenerationSettings(provider="sarvam")).generate_structured(SYSTEM_PROMPT, user, LLMAnswer)
    assert "300" in result.answer
