"""Evaluation runner (FR-08). OWNER: Member 1.

    python -m src.evaluation.run_eval            # full pipeline
    python -m src.evaluation.run_eval --ablation # BM25 vs dense vs hybrid (with Member 3)

Writes reports/eval_report.json and reports/eval_report.md (metrics table,
per-category breakdown, failure cases). Never edit the numbers by hand.
"""

from __future__ import annotations

from pathlib import Path

from src.pipeline import QAPipeline
from src.schemas import EvalQuestion, EvalReport


def load_questions(path: Path) -> list[EvalQuestion]:
    """Read data/eval/questions.jsonl (one EvalQuestion JSON per line)."""
    raise NotImplementedError("Member 1: implement load_questions")


def run_evaluation(pipeline: QAPipeline, questions: list[EvalQuestion], k: int = 5) -> EvalReport:
    """Ask every question, compute per-question metrics, aggregate overall + per category.
    A crash on one question is recorded in EvalQuestionResult.error, not raised."""
    raise NotImplementedError("Member 1: implement run_evaluation")


def write_report(report: EvalReport, reports_dir: Path) -> tuple[Path, Path]:
    """Write JSON + Markdown report; return both paths."""
    raise NotImplementedError("Member 1: implement write_report")


def main() -> None:
    raise NotImplementedError("Member 1: implement CLI entry point")


if __name__ == "__main__":
    main()
