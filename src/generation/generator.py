"""Grounded answer generation (FR-04, FR-05). OWNER: Member 4.

    question + passages -> prompt -> LLM (structured JSON) -> citations -> verification
    -> GeneratedAnswer

Unsupported claims are REMOVED from the final answer text (or the whole answer is
marked insufficient_evidence if nothing survives). Fall back to ExtractiveClient
when the LLM raises LLMError.
"""

from __future__ import annotations

from src.config import GenerationSettings
from src.generation.llm_client import LLMClient
from src.schemas import GeneratedAnswer, RetrievedPassage


class AnswerGenerator:
    def __init__(self, settings: GenerationSettings, client: LLMClient | None = None) -> None:
        self.settings = settings
        self.client = client  # None -> get_llm_client(settings) on first use

    def generate(self, question: str, passages: list[RetrievedPassage],
                 version: str | None = None) -> GeneratedAnswer:
        """Return a GeneratedAnswer whose citations reference only ``passages``.
        With zero passages, return insufficient_evidence=True without calling the LLM."""
        raise NotImplementedError("Member 4: implement AnswerGenerator.generate")
