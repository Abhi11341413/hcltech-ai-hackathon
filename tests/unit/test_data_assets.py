"""Sanity checks for the committed corpus and labelled evaluation set."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from src.schemas import AnswerStatus, EvalQuestion, QuestionCategory

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "data" / "sample_docs"
QUESTIONS = ROOT / "data" / "eval" / "questions.jsonl"


def _slug(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")


def _questions() -> list[EvalQuestion]:
    lines = QUESTIONS.read_text(encoding="utf-8").splitlines()
    return [EvalQuestion.model_validate(json.loads(line)) for line in lines if line.strip()]


def test_questions_validate_and_ids_unique():
    qs = _questions()
    assert len(qs) >= 25
    assert len({q.qid for q in qs}) == len(qs)


def test_every_category_is_covered():
    counts = Counter(q.category for q in _questions())
    assert all(counts[c] >= 3 for c in QuestionCategory)


def test_labels_point_to_existing_documents():
    doc_ids = {_slug(p) for p in DOCS.iterdir() if p.is_file()}
    for q in _questions():
        for label in q.relevant:
            assert label.doc_id in doc_ids, f"{q.qid}: unknown doc_id {label.doc_id}"


def test_status_matches_category():
    for q in _questions():
        if q.category in {QuestionCategory.AMBIGUOUS, QuestionCategory.UNANSWERABLE}:
            assert not q.relevant
        if q.category == QuestionCategory.CONFLICT:
            assert q.expected_status == AnswerStatus.CONFLICT and len(q.relevant) >= 2


def test_text_documents_are_marked_synthetic():
    for p in DOCS.iterdir():
        if p.suffix in {".md", ".txt", ".html"}:
            assert "SYNTHETIC" in p.read_text(encoding="utf-8"), p.name
