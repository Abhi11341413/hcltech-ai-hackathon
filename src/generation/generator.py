"""Grounded answer generation (FR-04, FR-05). OWNER: Member 4.

    question + passages -> prompt -> LLM (structured JSON) -> citations -> verification
    -> GeneratedAnswer

Unsupported claims are REMOVED from the final answer text (or the whole answer is
marked insufficient_evidence if nothing survives). Fall back to ExtractiveClient
when the LLM raises LLMError.
"""

from __future__ import annotations

import re

from src.config import GenerationSettings, get_logger
from src.errors import LLMError
from src.generation.citations import build_citations
from src.generation.llm_client import INSUFFICIENT_ANSWER, ExtractiveClient, LLMClient, get_llm_client
from src.generation.prompts import SYSTEM_PROMPT, LLMAnswer, build_user_prompt
from src.generation.verification import groundedness_score, strip_citation_markers, verify_claims
from src.schemas import Citation, Claim, GeneratedAnswer, RetrievedPassage

logger = get_logger(__name__)

SUPPORT_THRESHOLD = 0.6
_MARKER_RE = re.compile(r"\s*\[(\d+(?:\s*,\s*\d+)*)\]")


def _dedupe(ids: list[int]) -> list[int]:
    return list(dict.fromkeys(ids))


def _renumber_markers(text: str, mapping: dict[int, int]) -> str:
    """Rewrite '[2]' / '[1, 3]' markers through ``mapping``; markers that map to nothing are removed."""
    def repl(m: re.Match[str]) -> str:
        new = _dedupe([mapping[int(n)] for n in re.split(r"\s*,\s*", m.group(1)) if int(n) in mapping])
        return " " + "".join(f"[{n}]" for n in new) if new else ""
    return _MARKER_RE.sub(repl, text).strip()


class AnswerGenerator:
    def __init__(self, settings: GenerationSettings, client: LLMClient | None = None) -> None:
        self.settings = settings
        self.client = client  # None -> get_llm_client(settings) on first use

    def _get_client(self) -> LLMClient:
        if self.client is None:
            self.client = get_llm_client(self.settings)
        return self.client

    def _ask_llm(self, user: str) -> tuple[LLMAnswer, str]:
        """Call the configured client; on LLMError fall back to the extractive client."""
        client = self._get_client()
        try:
            return client.generate_structured(SYSTEM_PROMPT, user, LLMAnswer), client.model_name
        except LLMError as exc:
            if isinstance(client, ExtractiveClient):
                raise
            logger.warning("LLM call failed (%s: %s); falling back to extractive answer",
                           exc.error_code, exc.message)
            fallback = ExtractiveClient()
            return fallback.generate_structured(SYSTEM_PROMPT, user, LLMAnswer), fallback.model_name

    def generate(self, question: str, passages: list[RetrievedPassage],
                 version: str | None = None) -> GeneratedAnswer:
        """Return a GeneratedAnswer whose citations reference only ``passages``.
        With zero passages, return insufficient_evidence=True without calling the LLM.

        Steps: prompt the LLM with the top ``max_context_passages`` passages -> map its passage
        numbers to citations (dropping invented ones) -> verify every claim against its cited
        passages -> keep only supported claims and the citations they use, numbered 1..n.
        """
        if not passages:
            return GeneratedAnswer(answer_text=INSUFFICIENT_ANSWER, insufficient_evidence=True,
                                   model=self._get_client().model_name)

        context = passages[: self.settings.max_context_passages]
        llm, model = self._ask_llm(build_user_prompt(question, context, version))

        if llm.insufficient_evidence or not llm.claims:
            return GeneratedAnswer(answer_text=INSUFFICIENT_ANSWER, insufficient_evidence=True,
                                   llm_reported_conflict=llm.conflict, model=model)

        # 1. Provisional citations: passage number (as the LLM saw it) -> citation id.
        claim_texts = [strip_citation_markers(c.text) for c in llm.claims]  # LLMs often embed [n]
        citations = build_citations([(t, c.passage_numbers) for t, c in zip(claim_texts, llm.claims, strict=True)],
                                    context)
        cid_of_chunk = {c.chunk_id: c.citation_id for c in citations}
        number_to_cid = {n: cid_of_chunk[p.chunk.chunk_id]
                         for n, p in enumerate(context, start=1) if p.chunk.chunk_id in cid_of_chunk}
        claims = [Claim(text=t, citation_ids=_dedupe([number_to_cid[n] for n in c.passage_numbers
                                                      if n in number_to_cid]))
                  for t, c in zip(claim_texts, llm.claims, strict=True)]

        # 2. Verify each claim against the passages it cites.
        claims = verify_claims(claims, citations, context, threshold=SUPPORT_THRESHOLD)
        grounded = groundedness_score(claims)
        supported = [c for c in claims if c.supported]
        if not supported:
            logger.info("no claim survived verification (%d claims); reporting insufficient evidence",
                        len(claims))
            return GeneratedAnswer(answer_text=INSUFFICIENT_ANSWER, claims=claims,
                                   insufficient_evidence=True, llm_reported_conflict=llm.conflict,
                                   model=model, groundedness_score=grounded)

        # 3. Final answer text.
        if len(supported) == len(claims):
            answer_text = _renumber_markers(llm.answer, number_to_cid)
        else:
            dropped = len(claims) - len(supported)
            logger.info("removed %d unsupported claim(s) from the answer", dropped)
            answer_text = " ".join(
                f"{c.text}{''.join(f' [{i}]' for i in c.citation_ids)}" for c in supported
            )

        # 4. Keep only citations used by supported claims, renumbered 1..n in order of use.
        used = _dedupe([i for c in supported for i in c.citation_ids])
        final_id = {old: new for new, old in enumerate(used, start=1)}
        by_id = {c.citation_id: c for c in citations}
        final_citations: list[Citation] = [
            by_id[old].model_copy(update={"citation_id": final_id[old]}) for old in used
        ]
        final_claims = [c.model_copy(update={"citation_ids": [final_id[i] for i in c.citation_ids
                                                               if i in final_id]})
                        for c in claims]
        answer_text = _renumber_markers(answer_text, final_id) or INSUFFICIENT_ANSWER

        return GeneratedAnswer(
            answer_text=answer_text,
            claims=final_claims,
            citations=final_citations,
            insufficient_evidence=False,
            llm_reported_conflict=llm.conflict,
            model=model,
            groundedness_score=grounded,
        )
