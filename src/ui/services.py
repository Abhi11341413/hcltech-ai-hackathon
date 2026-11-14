"""UI service layer: everything the Streamlit page needs, with no Streamlit imports.

Kept separate from ``app.py`` so it can be unit-tested and so the page stays a thin view.

Key idea: the UI must work at every stage of the hackathon.

* ``check_readiness`` probes each member's module with a cheap call. A module whose
  functions still raise ``NotImplementedError`` is reported as "not merged yet".
* ``build_pipeline`` builds the real ``QAPipeline`` once ingestion, retrieval and
  reasoning are ready. If Member 4's generator is not ready yet, a clearly-labelled
  ``StandInGenerator`` (verbatim extractive answers) is injected so the rest of the
  pipeline can still be demonstrated end to end.
* ``demo_response`` returns canned mock responses (``src/mocks.py``) when the live
  pipeline cannot run at all, so the interface itself can always be shown.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from src import mocks
from src.config import PROJECT_ROOT, Settings, get_logger
from src.errors import UnsupportedFormatError, ValidationFailedError
from src.schemas import (
    AnswerStatus,
    Citation,
    Claim,
    ConfidenceAssessment,
    ConfidenceLevel,
    GeneratedAnswer,
    QAResponse,
    RetrievedPassage,
)

logger = get_logger(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
SAMPLE_DOCS_DIR = PROJECT_ROOT / "data" / "sample_docs"
STAND_IN_MODEL = "ui-stand-in-extractive"

EXAMPLE_QUESTIONS: dict[str, str] = {
    "Direct": "How do I reset my password?",
    "Paraphrase": "I forgot my login credentials, what should I do?",
    "Conflict": "How many API requests can I make per minute?",
    "Ambiguous": "What is the limit?",
    "Unanswerable": "Does the product support on-premise Kubernetes deployment?",
}


# --------------------------------------------------------------------------- readiness
@dataclass(frozen=True)
class ModuleStatus:
    key: str
    label: str
    owner: str
    ready: bool
    detail: str = ""


def _probe(calls: list[Callable[[], object]]) -> tuple[bool, str]:
    """A module is ready when none of its probe calls raise NotImplementedError.

    Any other exception (e.g. a validation error from a deliberately tiny input)
    still proves the function is implemented.
    """
    for call in calls:
        try:
            call()
        except NotImplementedError as exc:
            return False, str(exc) or "not implemented"
        except Exception:  # noqa: BLE001 - implemented but rejected the probe input
            continue
    return True, ""


def _ingestion_probes() -> list[Callable[[], object]]:
    from src.ingestion import chunking, cleaning, loaders
    return [lambda: loaders.detect_file_type(Path("probe.md")),
            lambda: cleaning.clean_text("a  b"), lambda: chunking.count_tokens("a b")]


def _retrieval_probes() -> list[Callable[[], object]]:
    from src.retrieval import hybrid, query_processing
    return [lambda: hybrid.reciprocal_rank_fusion([["a"]]), lambda: query_processing.tokenize("a")]


def _generation_probes() -> list[Callable[[], object]]:
    from src.generation import prompts, verification
    return [lambda: verification.groundedness_score([]), lambda: prompts.format_passages([])]


def _reasoning_probes() -> list[Callable[[], object]]:
    from src.reasoning import conflicts
    return [lambda: conflicts.extract_quantities("up to 300 requests per minute")]


def _evaluation_probes() -> list[Callable[[], object]]:
    from src.evaluation import metrics
    return [lambda: metrics.status_accuracy([AnswerStatus.ANSWERED], [AnswerStatus.ANSWERED])]


MODULES: list[tuple[str, str, str, Callable[[], list[Callable[[], object]]]]] = [
    ("ingestion", "Ingestion & chunking", "Member 2", _ingestion_probes),
    ("retrieval", "Hybrid retrieval", "Member 3", _retrieval_probes),
    ("generation", "Grounded generation", "Member 4", _generation_probes),
    ("reasoning", "Conflicts & confidence", "Member 1", _reasoning_probes),
    ("evaluation", "Evaluation", "Member 1", _evaluation_probes),
]


def check_readiness() -> dict[str, ModuleStatus]:
    statuses: dict[str, ModuleStatus] = {}
    for key, label, owner, probes in MODULES:
        try:
            ready, detail = _probe(probes())
        except ImportError as exc:
            ready, detail = False, f"import failed: {exc}"
        statuses[key] = ModuleStatus(key=key, label=label, owner=owner, ready=ready, detail=detail)
    return statuses


def live_mode_available(statuses: dict[str, ModuleStatus]) -> bool:
    """Live pipeline needs real ingestion, retrieval and reasoning (generation can be stood in)."""
    return all(statuses[k].ready for k in ("ingestion", "retrieval", "reasoning"))


def missing_owner(exc: NotImplementedError) -> str:
    """Turn 'Member 4: implement X' into 'Member 4'."""
    match = re.search(r"Member \d", str(exc))
    return match.group(0) if match else "a team member"


# --------------------------------------------------------------------------- settings
def has_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def effective_settings(base: Settings, prefer_offline: bool = False) -> tuple[Settings, str]:
    """Copy of settings with the LLM provider resolved. Returns (settings, explanation)."""
    settings = base.model_copy(deep=True)
    if prefer_offline:
        settings.generation.provider = "extractive"
        return settings, "Offline mode: answers are built from passage text without calling an LLM."
    if settings.generation.provider == "anthropic" and not has_api_key():
        settings.generation.provider = "extractive"
        return settings, "No ANTHROPIC_API_KEY found in .env, so the offline extractive generator is used."
    if settings.generation.provider == "anthropic":
        return settings, f"Answers are written by {settings.generation.model} using only retrieved passages."
    return settings, "Offline extractive generator (provider=extractive)."


# --------------------------------------------------------------------------- stand-in generator
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")


def _overlap(a: str, b: str) -> int:
    return len(set(_WORD_RE.findall(a.lower())) & set(_WORD_RE.findall(b.lower())))


class StandInGenerator:
    """Temporary replacement for Member 4's AnswerGenerator, used only by the UI.

    Picks the best-matching sentence from each of the top passages and cites it.
    Every claim is a verbatim excerpt, so groundedness is 1.0 by construction.
    It is clearly labelled in the UI and disappears once Member 4's module is merged.
    """

    model_name = STAND_IN_MODEL

    def __init__(self, max_passages: int = 2, min_score: float = 0.0) -> None:
        self.max_passages = max_passages
        self.min_score = min_score

    def generate(self, question: str, passages: list[RetrievedPassage],
                 version: str | None = None) -> GeneratedAnswer:
        usable = [p for p in passages if p.score >= self.min_score][: self.max_passages]
        if not usable:
            return GeneratedAnswer(answer_text="", insufficient_evidence=True, model=STAND_IN_MODEL)
        claims: list[Claim] = []
        citations: list[Citation] = []
        parts: list[str] = []
        for number, passage in enumerate(usable, start=1):
            sentences = [s.strip() for s in _SENTENCE_RE.split(passage.chunk.text) if s.strip()]
            best = max(sentences, key=lambda s: _overlap(s, question)) if sentences else passage.chunk.text
            chunk = passage.chunk
            citations.append(Citation(citation_id=number, chunk_id=chunk.chunk_id, doc_id=chunk.doc_id,
                                      doc_title=chunk.doc_title, section=chunk.section, page=chunk.page,
                                      source=chunk.source, excerpt=best))
            claims.append(Claim(text=best, citation_ids=[number], supported=True, support_score=1.0))
            parts.append(f"{best} [{number}]")
        return GeneratedAnswer(answer_text=" ".join(parts), claims=claims, citations=citations,
                               model=STAND_IN_MODEL, groundedness_score=1.0)


# --------------------------------------------------------------------------- pipeline
@dataclass
class PipelineBundle:
    pipeline: object  # QAPipeline (typed loosely to keep imports lazy)
    generator_label: str
    notes: list[str] = field(default_factory=list)


def build_pipeline(settings: Settings, statuses: dict[str, ModuleStatus]) -> PipelineBundle:
    """Build the real QAPipeline, standing in for Member 4 if their module is not merged yet.

    Raises:
        ValidationFailedError: ingestion, retrieval or reasoning is not ready.
    """
    if not live_mode_available(statuses):
        missing = [f"{s.label} ({s.owner})" for k, s in statuses.items()
                   if k in ("ingestion", "retrieval", "reasoning") and not s.ready]
        raise ValidationFailedError("Live pipeline needs: " + ", ".join(missing))
    from src.pipeline import QAPipeline

    notes: list[str] = []
    if statuses["generation"].ready:
        pipeline = QAPipeline(settings)
        label = settings.generation.model if settings.generation.provider == "anthropic" else "extractive (offline)"
    else:
        pipeline = QAPipeline(settings, generator=StandInGenerator())
        label = "UI stand-in (Member 4's generator not merged yet)"
        notes.append("Answers are verbatim sentences from the top passages until Member 4's generator is merged.")
    return PipelineBundle(pipeline=pipeline, generator_label=label, notes=notes)


# --------------------------------------------------------------------------- uploads
def sanitize_filename(name: str) -> str:
    """Strip directories and unsafe characters: '../../evil name.md' -> 'evil_name.md'."""
    base = Path(name.replace("\\", "/")).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    if not cleaned:
        raise ValidationFailedError("file name is empty after sanitising")
    return cleaned


def save_upload(name: str, data: bytes, settings: Settings, upload_dir: Path = UPLOAD_DIR) -> Path:
    """Validate and store an uploaded file under data/uploads/ (git-ignored).

    Raises:
        UnsupportedFormatError: extension not in settings.ingestion.supported_extensions.
        ValidationFailedError: empty or larger than MAX_UPLOAD_BYTES.
    """
    safe = sanitize_filename(name)
    suffix = Path(safe).suffix.lower()
    if suffix not in settings.ingestion.supported_extensions:
        raise UnsupportedFormatError(f"{safe}: unsupported format {suffix or '(none)'}")
    if not data:
        raise ValidationFailedError(f"{safe}: file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValidationFailedError(f"{safe}: larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / safe
    target.write_bytes(data)
    return target


# --------------------------------------------------------------------------- demo mode
def demo_response(question: str) -> QAResponse:
    """Canned response from src/mocks.py, chosen by simple keyword matching. UI preview only."""
    q = question.lower()
    base = mocks.mock_qa_response()
    passages = mocks.mock_passages()
    if "password" in q or "credential" in q or "login" in q or "log in" in q:
        chunk = passages[2].chunk
        citation = Citation(citation_id=1, chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, doc_title=chunk.doc_title,
                            section=chunk.section, source=chunk.source,
                            excerpt="open Settings > Security and choose Reset password.")
        return QAResponse(
            question=question, status=AnswerStatus.ANSWERED,
            answer="Open Settings > Security and choose Reset password; the emailed link is valid for 30 minutes [1].",
            citations=[citation], confidence=ConfidenceAssessment(
                score=0.84, level=ConfidenceLevel.HIGH, reasons=["strong match", "single consistent source"]),
            retrieved=[passages[2].model_copy(update={"rank": 1, "score": 0.81})], latency_ms=640.0)
    if re.search(r"\b(limit|quota)\b", q) and not re.search(r"request|rate|api|minute", q):
        return QAResponse(
            question=question, status=AnswerStatus.NEEDS_CLARIFICATION,
            answer="Do you mean the API rate limit, the seat limit or the upload size limit?",
            clarifying_question="Do you mean the API rate limit, the seat limit or the upload size limit?",
            confidence=ConfidenceAssessment(score=0.3, level=ConfidenceLevel.LOW,
                                            reasons=["question matches several unrelated topics"]),
            retrieved=passages, latency_ms=210.0)
    if re.search(r"request|rate|per minute|api", q):
        return base.model_copy(update={"question": question})
    from src.reasoning.confidence import NOT_FOUND_MESSAGE
    return QAResponse(
        question=question, status=AnswerStatus.NOT_FOUND, answer=NOT_FOUND_MESSAGE,
        confidence=ConfidenceAssessment(score=0.12, level=ConfidenceLevel.LOW,
                                        reasons=["no passage is relevant enough"]),
        retrieved=[], latency_ms=180.0)


STATUS_STYLE: dict[AnswerStatus, tuple[str, str]] = {
    AnswerStatus.ANSWERED: ("Answered", "green"),
    AnswerStatus.CONFLICT: ("Sources disagree", "orange"),
    AnswerStatus.NEEDS_CLARIFICATION: ("Needs clarification", "blue"),
    AnswerStatus.NOT_FOUND: ("Not found in the documents", "gray"),
}
