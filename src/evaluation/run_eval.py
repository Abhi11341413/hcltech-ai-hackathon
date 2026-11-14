"""Evaluation runner (FR-08). OWNER: Member 1.

    python -m src.evaluation.run_eval                          # full pipeline (provider from settings)
    python -m src.evaluation.run_eval --provider extractive    # offline, no API key
    python -m src.evaluation.run_eval --ablation               # + BM25 vs dense vs hybrid (with Member 3)

Writes reports/eval_report.json and reports/eval_report.md (metrics table,
per-category breakdown, failure cases). Never edit the numbers by hand.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Iterable
from pathlib import Path

from src.config import Settings, get_logger, get_settings
from src.errors import AssistantError, ValidationFailedError
from src.evaluation import metrics
from src.pipeline import QAPipeline, Retriever
from src.schemas import (
    AnswerStatus,
    Chunk,
    EvalQuestion,
    EvalQuestionResult,
    EvalReport,
    QueryRequest,
    QuestionCategory,
    RetrievedPassage,
    RetrieverName,
)

logger = get_logger(__name__)

SHOWN_ANSWER = {AnswerStatus.ANSWERED, AnswerStatus.CONFLICT}
FALLBACK_CATEGORIES = {QuestionCategory.AMBIGUOUS, QuestionCategory.UNANSWERABLE}


# --------------------------------------------------------------------------- loading
def load_questions(path: Path) -> list[EvalQuestion]:
    """Read data/eval/questions.jsonl (one EvalQuestion JSON per line; blank lines skipped).

    Raises:
        ValidationFailedError: file missing, a line is not a valid EvalQuestion, or qids repeat.
    """
    path = Path(path)
    if not path.is_file():
        raise ValidationFailedError(f"questions file not found: {path.name}")
    questions: list[EvalQuestion] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            questions.append(EvalQuestion.model_validate_json(line))
        except ValueError as exc:  # pydantic.ValidationError is a ValueError
            raise ValidationFailedError(f"{path.name} line {line_no}: invalid question ({exc})") from exc
    qids = [q.qid for q in questions]
    if len(qids) != len(set(qids)):
        raise ValidationFailedError(f"{path.name}: duplicate qids")
    return questions


# --------------------------------------------------------------------------- aggregation helpers
def _mean(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return round(sum(present) / len(present), 4) if present else None


def _percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile (no interpolation): simple to explain."""
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)], 1)


def _correct(r: EvalQuestionResult) -> bool:
    """A crashed question counts as wrong."""
    return r.error is None and r.predicted_status == r.expected_status


def _share(results: list[EvalQuestionResult], hit: Callable[[EvalQuestionResult], bool]) -> float | None:
    return round(sum(1 for r in results if hit(r)) / len(results), 4) if results else None


def _aggregate(results: list[EvalQuestionResult], k: int) -> dict[str, float]:
    """Means over the questions where a metric applies; a metric with no data is omitted, not 0."""
    candidates: dict[str, float | None] = {
        "n": float(len(results)),
        "status_accuracy": _share(results, _correct),
        f"recall@{k}": _mean(r.recall_at_k for r in results),
        "mrr": _mean(r.reciprocal_rank for r in results),
        f"ndcg@{k}": _mean(r.ndcg_at_k for r in results),
        "groundedness": _mean(r.groundedness for r in results),
        "citation_precision": _mean(r.citation_precision for r in results),
        "latency_ms_mean": _mean(r.latency_ms for r in results),
        "latency_ms_p95": _percentile([r.latency_ms for r in results if r.latency_ms is not None], 95),
        "errors": float(sum(1 for r in results if r.error)),
    }
    return {name: value for name, value in candidates.items() if value is not None}


# --------------------------------------------------------------------------- evaluation
def _evaluate_one(pipeline: QAPipeline, q: EvalQuestion, k: int) -> EvalQuestionResult:
    response, generated = pipeline.ask_detailed(QueryRequest(question=q.question, top_k=k))
    result = EvalQuestionResult(qid=q.qid, category=q.category, expected_status=q.expected_status,
                                predicted_status=response.status, latency_ms=response.latency_ms)
    if q.relevant:
        result.recall_at_k = metrics.recall_at_k(response.retrieved, q.relevant, k)
        result.reciprocal_rank = metrics.reciprocal_rank(response.retrieved, q.relevant)
        result.ndcg_at_k = metrics.ndcg_at_k(response.retrieved, q.relevant, k)
    if response.status in SHOWN_ANSWER and generated is not None:
        # groundedness of answers the user actually sees (computed by Member 4's verifier)
        from src.reasoning.confidence import groundedness_of
        result.groundedness = groundedness_of(generated)
        if q.relevant and response.citations:
            result.citation_precision = metrics.citation_precision(response.citations, q.relevant)
    return result


def run_evaluation(pipeline: QAPipeline, questions: list[EvalQuestion], k: int = 5) -> EvalReport:
    """Ask every question, compute per-question metrics, aggregate overall + per category.
    A crash on one question is recorded in EvalQuestionResult.error, not raised."""
    results: list[EvalQuestionResult] = []
    for q in questions:
        try:
            results.append(_evaluate_one(pipeline, q, k))
        except Exception as exc:  # noqa: BLE001 - one failing question must not abort the run
            logger.warning("question %s failed: %s", q.qid, type(exc).__name__)
            code = exc.error_code if isinstance(exc, AssistantError) else type(exc).__name__
            results.append(EvalQuestionResult(qid=q.qid, category=q.category, expected_status=q.expected_status,
                                              predicted_status=AnswerStatus.NOT_FOUND,
                                              error=f"{code}: {exc}"[:300]))

    overall = _aggregate(results, k)
    conflict_qs = [r for r in results if r.category == QuestionCategory.CONFLICT]
    non_conflict = [r for r in results if r.expected_status != AnswerStatus.CONFLICT]
    fallback_qs = [r for r in results if r.category in FALLBACK_CATEGORIES]
    extras = {
        "conflict_detection_recall": _share(conflict_qs, lambda r: r.predicted_status == AnswerStatus.CONFLICT),
        "false_conflict_rate": _share(non_conflict, lambda r: r.predicted_status == AnswerStatus.CONFLICT),
        "fallback_accuracy": _share(fallback_qs, _correct),
    }
    overall.update({name: value for name, value in extras.items() if value is not None})

    per_category = {cat.value: _aggregate([r for r in results if r.category == cat], k)
                    for cat in QuestionCategory if any(r.category == cat for r in results)}

    s = pipeline.settings
    config: dict[str, str | int | float | bool] = {
        "k": k,
        "retriever": "hybrid (bm25 + dense, RRF)" + (" + rerank" if s.retrieval.use_reranker else ""),
        "embedding_model": s.retrieval.embedding_model,
        "candidate_pool": s.retrieval.candidate_pool,
        "rrf_k": s.retrieval.rrf_k,
        "generation_provider": s.generation.provider,
        "generation_model": s.generation.model,
        "min_top_score_for_answer": s.reasoning.min_top_score_for_answer,
        "min_groundedness": s.reasoning.min_groundedness,
        "chunk_max_tokens": s.ingestion.chunk_max_tokens,
    }
    return EvalReport(k=k, n_questions=len(questions), metrics=overall, per_category=per_category,
                      results=results, retriever_config=config)


# --------------------------------------------------------------------------- ablation (FR-08 + M3)
class SingleIndexRetriever:
    """Adapter so a lone BM25Index / DenseIndex can be evaluated like the hybrid retriever.

    Scores are normalised to [0, 1] only for display: BM25 divided by the best score of the
    query, cosine clipped at 0. Ranking metrics do not depend on this normalisation.
    """

    def __init__(self, name: RetrieverName, index: object, pool: int) -> None:
        self.name, self.index, self.pool = name, index, pool

    def build(self, chunks: list[Chunk]) -> None:
        self.index.build(chunks)  # type: ignore[attr-defined]

    def retrieve(self, question: str, top_k: int = 5, doc_ids: list[str] | None = None) -> list[RetrievedPassage]:
        hits = self.index.search(question, top_n=self.pool)  # type: ignore[attr-defined]
        hits = [(c, s) for c, s in hits if doc_ids is None or c.doc_id in doc_ids][:top_k]
        top = max((s for _, s in hits), default=0.0)
        return [RetrievedPassage(chunk=c, rank=i, retriever=self.name, score_breakdown={self.name.value: float(s)},
                                 score=max(0.0, min(1.0, s / top if self.name == RetrieverName.BM25 and top > 0
                                                    else s)))
                for i, (c, s) in enumerate(hits, start=1)]


def _ablation_retrievers(settings: Settings) -> dict[str, Retriever]:
    from src.retrieval.bm25_index import BM25Index
    from src.retrieval.dense_index import DenseIndex, Embedder
    from src.retrieval.hybrid import HybridRetriever

    pool = settings.retrieval.candidate_pool
    return {
        "bm25": SingleIndexRetriever(RetrieverName.BM25, BM25Index(), pool),
        "dense": SingleIndexRetriever(RetrieverName.DENSE,
                                      DenseIndex(Embedder(settings.retrieval.embedding_model)), pool),
        "hybrid": HybridRetriever(settings.retrieval),
    }


def run_ablation(chunks: list[Chunk], questions: list[EvalQuestion], k: int,
                 retrievers: dict[str, Retriever]) -> dict[str, dict[str, float]]:
    """Retrieval-only comparison on questions that have relevance labels (no LLM calls).

    Output: {retriever_name: {"recall@k", "mrr", "ndcg@k", "n"}}. A retriever that fails to
    build is reported with an "error" flag instead of aborting the run.
    """
    labelled = [q for q in questions if q.relevant]
    table: dict[str, dict[str, float]] = {}
    for name, retriever in retrievers.items():
        try:
            retriever.build(chunks)
            ranked = [(q, retriever.retrieve(q.question, top_k=k)) for q in labelled]
        except Exception as exc:  # noqa: BLE001 - report, don't abort
            logger.warning("ablation retriever %s failed: %s", name, type(exc).__name__)
            table[name] = {"error": 1.0}
            continue
        table[name] = {
            "n": float(len(labelled)),
            f"recall@{k}": _mean(metrics.recall_at_k(p, q.relevant, k) for q, p in ranked) or 0.0,
            "mrr": _mean(metrics.reciprocal_rank(p, q.relevant) for q, p in ranked) or 0.0,
            f"ndcg@{k}": _mean(metrics.ndcg_at_k(p, q.relevant, k) for q, p in ranked) or 0.0,
        }
    return table


# --------------------------------------------------------------------------- reporting
_METRIC_HELP = {
    "status_accuracy": "share of questions whose status (answered/conflict/clarify/not found) matches the label",
    "recall@k": "share of gold doc/sections found in the top k (questions with labels only)",
    "mrr": "mean of 1 / rank of the first relevant passage",
    "ndcg@k": "graded ranking quality, 1.0 = ideal order",
    "groundedness": "share of claims supported by cited passages, for answers actually shown",
    "citation_precision": "share of shown citations that point to a gold doc/section",
    "conflict_detection_recall": "conflict questions answered with status=conflict",
    "false_conflict_rate": "non-conflict questions wrongly flagged as conflict",
    "fallback_accuracy": "ambiguous + unanswerable questions given the right fallback",
    "latency_ms_mean": "mean end-to-end latency of ask()",
    "latency_ms_p95": "95th percentile latency (nearest rank)",
    "errors": "questions that crashed (counted as wrong)",
    "n": "number of questions",
}


def _fmt(value: float | None) -> str:
    if value is None:
        return "–"
    return f"{value:.0f}" if float(value).is_integer() and abs(value) >= 1 else f"{value:.3f}"


def _help(name: str) -> str:
    return _METRIC_HELP.get(name.split("@")[0] + ("@k" if "@" in name else ""), "")


def render_markdown(report: EvalReport, ablation: dict[str, dict[str, float]] | None = None,
                    questions: list[EvalQuestion] | None = None) -> str:
    """Render the report as Markdown. Pure function of its inputs (no numbers typed by hand)."""
    text = {q.qid: q.question for q in questions or []}
    k = report.k
    out = [
        "# Evaluation report — Enterprise Documentation Q&A Assistant",
        "",
        f"_Generated {report.created_at:%Y-%m-%d %H:%M UTC} by `python -m src.evaluation.run_eval`. "
        "Do not edit numbers by hand; re-run the command._",
        "",
        f"Questions: **{report.n_questions}** · k = {k} · labelled set: `data/eval/questions.jsonl` "
        "(synthetic, written by the team; see `data/README.md`).",
        "",
        "## Configuration",
        "",
        "| Setting | Value |", "|---|---|",
        *[f"| {key} | `{value}` |" for key, value in report.retriever_config.items()],
        "",
        "## Overall metrics",
        "",
        "| Metric | Value | Meaning |", "|---|---|---|",
        *[f"| {name} | {_fmt(value)} | {_help(name)} |" for name, value in report.metrics.items()],
        "",
        "## Per category",
        "",
    ]
    cols = ["n", "status_accuracy", f"recall@{k}", "mrr", f"ndcg@{k}", "groundedness", "citation_precision"]
    out += ["| Category | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for cat, values in report.per_category.items():
        out.append(f"| {cat} | " + " | ".join(_fmt(values.get(c)) for c in cols) + " |")

    if ablation:
        out += ["", "## Retrieval ablation (labelled questions only, no LLM)", "",
                "BM25 and dense are evaluated on the raw question; hybrid includes Member 3's query expansion "
                "and RRF fusion.", "",
                f"| Retriever | recall@{k} | MRR | nDCG@{k} |", "|---|---|---|---|"]
        for name, values in ablation.items():
            if "error" in values:
                out.append(f"| {name} | failed | failed | failed |")
            else:
                out.append(f"| {name} | {_fmt(values.get(f'recall@{k}'))} | {_fmt(values.get('mrr'))} | "
                           f"{_fmt(values.get(f'ndcg@{k}'))} |")

    out += ["", "## Per question", "",
            "| qid | category | expected | predicted | ok | recall | RR | grounded | latency ms |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in report.results:
        out.append(f"| {r.qid} | {r.category.value} | {r.expected_status.value} | {r.predicted_status.value} | "
                   f"{'✅' if _correct(r) else '❌'} | {_fmt(r.recall_at_k)} | {_fmt(r.reciprocal_rank)} | "
                   f"{_fmt(r.groundedness)} | {_fmt(r.latency_ms)} |")

    failures = [r for r in report.results
                if not _correct(r) or (r.recall_at_k is not None and r.recall_at_k < 1.0)]
    out += ["", "## Failure cases", ""]
    if not failures:
        out.append("No failures on this run.")
    for r in failures:
        why = []
        if r.error:
            why.append(f"error `{r.error}`")
        elif r.predicted_status != r.expected_status:
            why.append(f"expected **{r.expected_status.value}**, got **{r.predicted_status.value}**")
        if r.recall_at_k is not None and r.recall_at_k < 1.0:
            why.append(f"recall@{k} = {_fmt(r.recall_at_k)} (missed gold sources)")
        out.append(f"- **{r.qid}** ({r.category.value}) {text.get(r.qid, '')} — " + "; ".join(why))
    out += ["", "## How metrics are computed", "",
            "- Relevance is judged at doc + section level: a passage is relevant if its `doc_id` matches a label "
            "and its section starts with the label's section (PDF labels match any section).",
            "- Retrieval metrics use only questions with labels (ambiguous/unanswerable have none).",
            "- Groundedness and citation precision use only answers actually shown to the user "
            "(status answered or conflict).",
            "- A question that crashes is recorded with its error and counted as wrong.", ""]
    return "\n".join(out)


def write_report(report: EvalReport, reports_dir: Path,
                 ablation: dict[str, dict[str, float]] | None = None,
                 questions: list[EvalQuestion] | None = None) -> tuple[Path, Path]:
    """Write JSON + Markdown report; return both paths (eval_report.json, eval_report.md)."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = reports_dir / "eval_report.json", reports_dir / "eval_report.md"
    payload = report.model_dump(mode="json")
    if ablation:
        payload["ablation"] = ablation
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report, ablation, questions), encoding="utf-8")
    return json_path, md_path


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    settings = get_settings().model_copy(deep=True)
    parser = argparse.ArgumentParser(description="Evaluate the Q&A pipeline on the labelled question set.")
    parser.add_argument("--k", type=int, default=settings.evaluation.k)
    parser.add_argument("--questions", default=settings.evaluation.questions_path)
    parser.add_argument("--data-dir", default=settings.data_dir)
    parser.add_argument("--reports-dir", default=settings.evaluation.reports_dir)
    parser.add_argument("--provider", choices=["anthropic", "extractive"],
                        help="override generation.provider (extractive = offline, no API key)")
    parser.add_argument("--ablation", action="store_true", help="also compare BM25 vs dense vs hybrid")
    args = parser.parse_args(argv)
    if args.provider:
        settings.generation.provider = args.provider

    questions = load_questions(settings.resolve(args.questions))
    pipeline = QAPipeline(settings)
    ingest = pipeline.ingest_directory(settings.resolve(args.data_dir))
    print(f"Ingested {ingest.documents_ingested} documents, {ingest.chunks_created} chunks, "
          f"{len(ingest.failures)} failures")

    report = run_evaluation(pipeline, questions, k=args.k)
    ablation = run_ablation(pipeline.all_chunks(), questions, args.k, _ablation_retrievers(settings)) \
        if args.ablation else None
    json_path, md_path = write_report(report, settings.resolve(args.reports_dir), ablation, questions)
    for name, value in report.metrics.items():
        print(f"  {name:28s} {_fmt(value)}")
    print(f"Report written to {md_path} and {json_path}")


if __name__ == "__main__":
    main()
