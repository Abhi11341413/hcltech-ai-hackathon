"""LLM access layer. OWNER: Member 4.

Two implementations behind one interface so the demo never dies:

* ``AnthropicClient`` - calls Claude via the official ``anthropic`` SDK.
  Key comes from ANTHROPIC_API_KEY in the environment (never hardcode it).
  Default model from settings: ``claude-opus-5-5``. Notes for the implementer:
    - Use ``client.messages.parse(...)`` / ``output_config={"format": ...}`` with a
      Pydantic model to get validated JSON (no assistant prefill - it returns 400).
    - Do NOT send ``temperature``/``top_p`` or ``thinking={"type": "disabled"}``
      (both rejected on this model); control cost/latency with
      ``output_config={"effort": "low"}``.
    - Check ``response.stop_reason`` ("refusal", "max_tokens") before reading content.
    - Catch ``anthropic.RateLimitError`` / ``APIConnectionError`` / ``APIStatusError``
      separately and re-raise as ``src.errors.LLMError``.
* ``GeminiClient`` - calls Google Gemini via ``google-genai`` (provider="gemini",
  key from GEMINI_API_KEY). Same structured-output contract as AnthropicClient.
* ``SarvamClient`` (``sarvam_client.py``) - calls Sarvam AI ``sarvam-105b`` over its REST
  API (provider="sarvam", key from SARVAM_API_KEY). Same structured-output contract.
* ``ExtractiveClient`` - no network: builds an answer from the highest-scoring
  sentences of the passages. Used when provider="extractive" or no API key.
"""

from __future__ import annotations

import os
import re
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from src.config import GenerationSettings, get_logger
from src.errors import ConfigError, LLMError, LLMOutputParseError
from src.generation.prompts import (
    PASSAGES_CLOSE,
    PASSAGES_OPEN,
    QUESTION_CLOSE,
    QUESTION_OPEN,
    LLMAnswer,
    LLMClaim,
)
from src.reasoning.text_utils import content_terms, split_sentences

T = TypeVar("T", bound=BaseModel)

logger = get_logger(__name__)

INSUFFICIENT_ANSWER = "The provided documents do not contain enough information to answer this question."


class LLMClient(Protocol):
    model_name: str

    def generate_structured(self, system: str, user: str, schema: type[T]) -> T:
        """Return an instance of ``schema`` parsed from the model output.

        Raises:
            LLMError: transport/API/refusal failure.
            LLMOutputParseError: output did not validate against ``schema``.
        """
        ...


class AnthropicClient:
    """Claude via ``client.messages.parse`` with the answer schema as structured output."""

    def __init__(self, settings: GenerationSettings) -> None:
        self.settings = settings
        self.model_name = settings.model
        self._client: Any = None  # anthropic.Anthropic, created on first call

    def _sdk(self) -> Any:
        if self._client is None:
            import anthropic  # local import: unit tests and the extractive path never need it

            # Credentials are resolved by the SDK from the environment (ANTHROPIC_API_KEY).
            self._client = anthropic.Anthropic(timeout=self.settings.timeout_seconds, max_retries=2)
        return self._client

    def generate_structured(self, system: str, user: str, schema: type[T]) -> T:
        """Call Claude once and return the validated ``schema`` instance.

        Raises:
            LLMError: rate limit, network, API status error or refusal.
            LLMOutputParseError: output truncated (max_tokens) or not valid for ``schema``.
        """
        import anthropic

        try:
            response = self._sdk().messages.parse(
                model=self.settings.model,
                max_tokens=self.settings.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
                output_config={"effort": self.settings.effort},
            )
        except anthropic.RateLimitError as exc:
            raise LLMError("LLM rate limit reached", details={"status": "429"}) from exc
        except anthropic.APIConnectionError as exc:  # includes APITimeoutError
            raise LLMError(f"could not reach the LLM API: {type(exc).__name__}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"LLM API returned HTTP {exc.status_code}",
                           details={"status": str(exc.status_code)}) from exc
        except (ValidationError, ValueError) as exc:
            raise LLMOutputParseError(f"LLM output did not match {schema.__name__}") from exc

        if response.stop_reason == "refusal":
            raise LLMError("the model declined to answer (refusal)")
        if response.stop_reason == "max_tokens":
            raise LLMOutputParseError("LLM output was cut off at max_tokens")
        parsed = response.parsed_output
        if parsed is None:
            raise LLMOutputParseError(f"LLM returned no {schema.__name__} object")
        logger.info("LLM answer from %s (stop_reason=%s, input_tokens=%s, output_tokens=%s)",
                    self.model_name, response.stop_reason,
                    getattr(response.usage, "input_tokens", "?"), getattr(response.usage, "output_tokens", "?"))
        return parsed


DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"  # fast and available; gemini-2.5-* is 404 for new keys
_GEMINI_BAD_FINISH = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "OTHER"}


class GeminiClient:
    """Google Gemini via the ``google-genai`` SDK, with ``LLMAnswer`` as the JSON response schema.

    Key comes from GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment / ``.env``.
    The model is ``generation.model`` when it names a Gemini model, else ``gemini-3.5-flash-lite``,
    so switching provider does not require editing the shared ``config/settings.yaml``.
    """

    def __init__(self, settings: GenerationSettings) -> None:
        self.settings = settings
        self.model_name = settings.model if settings.model.startswith("gemini") else DEFAULT_GEMINI_MODEL
        self._client: Any = None  # google.genai.Client, created on first call

    def _sdk(self) -> Any:
        if self._client is None:
            from google import genai  # local import: only needed when provider == "gemini"
            from google.genai import types

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            self._client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=int(self.settings.timeout_seconds * 1000)),
            )
        return self._client

    def generate_structured(self, system: str, user: str, schema: type[T]) -> T:
        """Call Gemini once and return the validated ``schema`` instance.

        Raises:
            LLMError: API/network error or a safety block.
            LLMOutputParseError: output truncated (MAX_TOKENS) or not valid for ``schema``.
        """
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            max_output_tokens=self.settings.max_tokens,
        )
        try:
            response = self._sdk().models.generate_content(model=self.model_name, contents=user, config=config)
        except errors.APIError as exc:  # ClientError (4xx, incl. 429) and ServerError (5xx)
            raise LLMError(f"Gemini API returned HTTP {exc.code}", details={"status": str(exc.code)}) from exc
        except OSError as exc:  # network failure / timeout below the SDK
            raise LLMError(f"could not reach the Gemini API: {type(exc).__name__}") from exc

        candidates = response.candidates or []
        finish = candidates[0].finish_reason if candidates else None
        finish_name = getattr(finish, "name", str(finish)) if finish is not None else "NONE"
        if not candidates or finish_name in _GEMINI_BAD_FINISH:
            raise LLMError(f"Gemini returned no usable answer (finish_reason={finish_name})")
        if finish_name == "MAX_TOKENS":
            raise LLMOutputParseError("Gemini output was cut off at max_output_tokens")

        parsed = response.parsed
        if isinstance(parsed, schema):
            result = parsed
        else:
            try:
                result = schema.model_validate_json(response.text or "")
            except (ValidationError, ValueError) as exc:
                raise LLMOutputParseError(f"Gemini output did not match {schema.__name__}") from exc
        logger.info("LLM answer from %s (finish_reason=%s)", self.model_name, finish_name)
        return result


_PASSAGE_HEADER_RE = re.compile(r"^\[(\d+)\] \(.*\)$", re.MULTILINE)
_QA_PREFIX_RE = re.compile(r"^[QA]:\s*")


def _between(text: str, start: str, end: str) -> str:
    i = text.find(start)
    j = text.rfind(end)
    return text[i + len(start):j] if i >= 0 and j > i else ""


def parse_user_prompt(user: str) -> tuple[str, list[tuple[int, str]]]:
    """Read back (question, [(passage_number, passage_text)]) from ``prompts.build_user_prompt`` output."""
    question = _between(user, QUESTION_OPEN, QUESTION_CLOSE).strip()
    block = _between(user, PASSAGES_OPEN, PASSAGES_CLOSE)
    headers = list(_PASSAGE_HEADER_RE.finditer(block))
    passages = []
    for k, m in enumerate(headers):
        end = headers[k + 1].start() if k + 1 < len(headers) else len(block)
        passages.append((int(m.group(1)), block[m.end():end].strip()))
    return question, passages


class ExtractiveClient:
    """Offline fallback: answers with the passage sentences that share the most terms with the question.

    Each chosen sentence is copied verbatim from a passage and cited, so verification always
    has something concrete to check. Conflict detection is left to Member 1's rules.
    """

    model_name = "extractive-fallback"
    max_sentences = 2
    min_overlap = 0.25  # fraction of the question's content terms a sentence must contain

    def generate_structured(self, system: str, user: str, schema: type[T]) -> T:
        if schema is not LLMAnswer:
            raise LLMOutputParseError(f"ExtractiveClient can only produce LLMAnswer, not {schema.__name__}")
        question, passages = parse_user_prompt(user)
        q_terms = content_terms(question)

        candidates: list[tuple[float, int, str]] = []  # (score, passage_number, sentence)
        for number, text in passages:
            best: tuple[float, int, str] | None = None
            for sentence in split_sentences(text):
                if sentence.endswith("?"):
                    continue  # FAQ questions restate the topic but answer nothing
                sentence = _QA_PREFIX_RE.sub("", sentence)
                score = len(q_terms & content_terms(sentence)) / len(q_terms) if q_terms else 0.0
                if score >= self.min_overlap and (best is None or score > best[0]):
                    best = (score, number, sentence)
            if best:
                candidates.append(best)

        chosen = sorted(candidates, key=lambda c: (-c[0], c[1]))[: self.max_sentences]
        if not chosen:
            return LLMAnswer(answer=INSUFFICIENT_ANSWER, claims=[], insufficient_evidence=True,
                             conflict=False)  # type: ignore[return-value]
        claims = [LLMClaim(text=sentence, passage_numbers=[number]) for _, number, sentence in chosen]
        answer = " ".join(f"{sentence} [{number}]" for _, number, sentence in chosen)
        return LLMAnswer(answer=answer, claims=claims, insufficient_evidence=False,
                         conflict=False)  # type: ignore[return-value]


PROVIDERS = ("anthropic", "gemini", "sarvam", "extractive")


def _has_anthropic_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _has_gemini_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def get_llm_client(settings: GenerationSettings) -> LLMClient:
    """Return the client for ``settings.provider`` when its key is configured.

    If the chosen provider has no key, fall back to the first other provider that has one
    (Anthropic -> Gemini -> Sarvam), so a machine with only one key in ``.env`` needs no
    settings change; with no key at all, use ExtractiveClient. Every fallback logs a WARNING.

    Raises:
        ConfigError: unknown provider name.
    """
    from src.generation.sarvam_client import SarvamClient, has_sarvam_key

    provider = settings.provider.strip().lower()
    if provider not in PROVIDERS:
        raise ConfigError(f"unknown generation.provider '{settings.provider}' (use {', '.join(PROVIDERS)})")
    if provider == "extractive":
        return ExtractiveClient()

    available = {"anthropic": _has_anthropic_key(), "gemini": _has_gemini_key(), "sarvam": has_sarvam_key()}
    build = {"anthropic": AnthropicClient, "gemini": GeminiClient, "sarvam": SarvamClient}
    if available[provider]:
        return build[provider](settings)
    for other in ("anthropic", "gemini", "sarvam"):
        if available[other]:
            logger.warning("no API key for provider '%s'; using '%s' instead", provider, other)
            return build[other](settings)
    logger.warning("no LLM API key configured (provider '%s'); using the offline extractive fallback", provider)
    return ExtractiveClient()
