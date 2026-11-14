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
* ``ExtractiveClient`` - no network: builds an answer from the highest-scoring
  sentences of the passages. Used when provider="extractive" or no API key.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

from src.config import GenerationSettings

T = TypeVar("T", bound=BaseModel)


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
    def __init__(self, settings: GenerationSettings) -> None:
        self.settings = settings
        self.model_name = settings.model

    def generate_structured(self, system: str, user: str, schema: type[T]) -> T:
        raise NotImplementedError("Member 4: implement AnthropicClient.generate_structured")


class ExtractiveClient:
    model_name = "extractive-fallback"

    def generate_structured(self, system: str, user: str, schema: type[T]) -> T:
        raise NotImplementedError("Member 4: implement ExtractiveClient.generate_structured")


def get_llm_client(settings: GenerationSettings) -> LLMClient:
    """Return AnthropicClient if provider == 'anthropic' and a key is configured,
    else ExtractiveClient (log a WARNING when falling back)."""
    raise NotImplementedError("Member 4: implement get_llm_client")
