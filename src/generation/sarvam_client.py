"""Sarvam AI chat-completions client (provider="sarvam"). Implements the ``LLMClient`` protocol.

Based on Sarvam's public API docs (https://docs.sarvam.ai, read 2026-11-16):

* ``POST https://api.sarvam.ai/v1/chat/completions`` (OpenAI-style request/response).
* Auth header ``api-subscription-key: <key>``. The key is read from ``SARVAM_API_KEY``
  (``.env``), never hardcoded and never logged.
* Model ``sarvam-105b`` (the flagship; ``sarvam-m`` and ``sarvam-30b`` are deprecated).
* Structured output via ``response_format = {"type": "json_schema", ...}``; the JSON comes
  back as a string in ``choices[0].message.content``. If the API rejects the schema
  (HTTP 400) we retry once in plain JSON mode (``{"type": "json_object"}``) with the
  schema written into the system prompt, then validate with Pydantic either way.
* Reasoning ("thinking") tokens count towards ``max_tokens`` and are returned separately
  in ``reasoning_content``; we only read ``content``. ``reasoning_effort`` comes from
  ``generation.effort`` (default "low") to keep latency down.
* Errors: 403 invalid key, 402 no credits, 404 unknown model, 429 rate limit,
  5xx overloaded. 429/5xx are retried once (honouring ``Retry-After``, capped).

We call the REST endpoint with the standard library (``urllib``) instead of the
``sarvamai`` SDK: no extra dependency, and every request field is visible here.
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from src.config import GenerationSettings, get_logger
from src.errors import LLMError, LLMOutputParseError

T = TypeVar("T", bound=BaseModel)

logger = get_logger(__name__)

SARVAM_CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"
DEFAULT_SARVAM_MODEL = "sarvam-105b"
SARVAM_KEY_ENV = "SARVAM_API_KEY"
_RETRYABLE = {429, 500, 502, 503}
_MAX_RETRY_WAIT_S = 10.0
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def has_sarvam_key() -> bool:
    return bool(os.environ.get(SARVAM_KEY_ENV, "").strip())


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON schema made friendly to strict structured-output engines:
    ``$ref``/``$defs`` inlined, every object closed (``additionalProperties: false``)
    and every property required. Titles/defaults are dropped (they are noise to the model)."""
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(copy.deepcopy(defs[node["$ref"].rsplit("/", 1)[-1]]))
            out = {k: resolve(v) for k, v in node.items() if k not in ("title", "default")}
            if out.get("type") == "object" and "properties" in out:
                out["additionalProperties"] = False
                out["required"] = list(out["properties"])
            return out
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(raw)


def extract_json_text(content: str) -> str:
    """Remove ``<think>`` blocks and Markdown code fences some models wrap around JSON."""
    text = _THINK_RE.sub("", content).strip()
    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    return text


class SarvamClient:
    """Sarvam AI (``sarvam-105b``) with the answer schema as structured JSON output."""

    def __init__(self, settings: GenerationSettings, api_key: str | None = None,
                 url: str = SARVAM_CHAT_URL) -> None:
        self.settings = settings
        self.model_name = settings.model if settings.model.startswith("sarvam") else DEFAULT_SARVAM_MODEL
        self._api_key = api_key
        self.url = url

    # ------------------------------------------------------------------ HTTP
    def _key(self) -> str:
        key = (self._api_key or os.environ.get(SARVAM_KEY_ENV, "")).strip()
        if not key:
            raise LLMError(f"{SARVAM_KEY_ENV} is not set; add it to .env")
        return key

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST once, retrying a single time on 429/5xx. Returns the decoded JSON body."""
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json", "api-subscription-key": self._key()}
        for attempt in (1, 2):
            request = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                status, code, message = exc.code, *_error_details(exc)
                if status in _RETRYABLE and attempt == 1:
                    wait = _retry_after(exc)
                    logger.warning("Sarvam HTTP %s (%s); retrying in %.1fs", status, code, wait)
                    time.sleep(wait)
                    continue
                raise LLMError(_friendly_error(status, code, message),
                               details={"status": str(status), "code": code}) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise LLMError(f"could not reach the Sarvam API: {type(exc).__name__}") from exc
            except json.JSONDecodeError as exc:
                raise LLMError("Sarvam API returned a non-JSON response") from exc
        raise LLMError("Sarvam API retry loop exhausted")  # pragma: no cover - loop always returns/raises

    # ------------------------------------------------------------------ protocol
    def _body(self, system: str, user: str, response_format: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": self.settings.max_tokens,
            "temperature": 0.2,
            "response_format": response_format,
        }
        effort = (self.settings.effort or "").strip().lower()
        if effort in ("none", "off", "disabled"):
            body["reasoning_effort"] = None
        elif effort in ("low", "medium", "high", "max"):
            body["reasoning_effort"] = effort
        return body

    def generate_structured(self, system: str, user: str, schema: type[T]) -> T:
        """Call Sarvam once (plus a JSON-mode retry if the schema is rejected) and validate.

        Raises:
            LLMError: auth/quota/rate-limit/network failure or a content-filter stop.
            LLMOutputParseError: output cut off at max_tokens, empty, or not valid for ``schema``.
        """
        schema_format = {"type": "json_schema", "json_schema": {
            "name": schema.__name__, "schema": strict_json_schema(schema), "strict": True}}
        try:
            payload = self._post(self._body(system, user, schema_format))
        except LLMError as exc:
            if exc.details.get("status") != "400":
                raise
            logger.warning("Sarvam rejected json_schema output (%s); retrying in JSON mode", exc.message)
            schema_text = json.dumps(strict_json_schema(schema))
            json_system = f"{system}\n\nReply with ONLY a JSON object matching this JSON Schema:\n{schema_text}"
            payload = self._post(self._body(json_system, user, {"type": "json_object"}))
        return self._parse(payload, schema)

    def _parse(self, payload: dict[str, Any], schema: type[T]) -> T:
        choices = payload.get("choices") or []
        if not choices:
            raise LLMOutputParseError("Sarvam returned no choices")
        choice = choices[0]
        finish = choice.get("finish_reason")
        message = choice.get("message") or {}
        if finish == "content_filter" or message.get("refusal"):
            raise LLMError("Sarvam declined to answer (content filter or refusal)")
        if finish == "length":
            raise LLMOutputParseError("Sarvam output was cut off at max_tokens (reasoning tokens count too)")
        content = message.get("content") or ""
        if not content.strip():
            raise LLMOutputParseError("Sarvam returned empty content")
        try:
            result = schema.model_validate_json(extract_json_text(content))
        except (ValidationError, ValueError) as exc:
            raise LLMOutputParseError(f"Sarvam output did not match {schema.__name__}") from exc
        usage = payload.get("usage") or {}
        logger.info("LLM answer from %s (finish_reason=%s, prompt_tokens=%s, completion_tokens=%s)",
                    self.model_name, finish, usage.get("prompt_tokens", "?"), usage.get("completion_tokens", "?"))
        return result


def _error_details(exc: urllib.error.HTTPError) -> tuple[str, str]:
    """(code, message) from Sarvam's ``{"error": {"code": ..., "message": ...}}`` body."""
    try:
        error = json.loads(exc.read().decode("utf-8")).get("error") or {}
        return str(error.get("code") or "unknown"), str(error.get("message") or "")
    except (ValueError, AttributeError, OSError):
        return "unknown", ""


def _retry_after(exc: urllib.error.HTTPError) -> float:
    try:
        return min(float(exc.headers.get("Retry-After", "2")), _MAX_RETRY_WAIT_S)
    except (TypeError, ValueError):
        return 2.0


def _friendly_error(status: int, code: str, message: str) -> str:
    hints = {
        403: "Sarvam rejected the API key (check SARVAM_API_KEY in .env)",
        402: "Sarvam account has no credits left",
        404: "Sarvam does not know this model id",
        429: "Sarvam rate limit reached",
    }
    base = hints.get(status, f"Sarvam API returned HTTP {status}")
    return f"{base} [{code}]" + (f": {message[:200]}" if message and status not in (402, 403) else "")
