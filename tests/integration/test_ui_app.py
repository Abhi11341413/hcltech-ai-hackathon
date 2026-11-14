"""Integration tests for the UI.

* ``AppTest`` runs the real Streamlit page headlessly and clicks through it.
* ``test_live_pipeline_with_stand_ins`` proves the UI's live path works end to end
  through Member 1's real pipeline and reasoning, with only the not-yet-merged
  pieces faked, so the UI will run unchanged once everyone's code is merged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import mocks
from src.config import Settings
from src.schemas import AnswerStatus, QueryRequest
from src.ui import services

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[2] / "src" / "ui" / "app.py")


def test_app_renders_without_errors():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Enterprise Documentation Q&A Assistant"
    assert len(at.tabs) == 3


def test_demo_conflict_question_shows_conflict_and_citations():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Demo (mock data)").run()
    at.text_input(key="question").set_value(services.EXAMPLE_QUESTIONS["Conflict"]).run()
    next(b for b in at.button if b.label == "Ask").click().run()
    assert not at.exception
    rendered = " ".join(m.value for m in at.markdown)
    assert "Sources disagree" in rendered
    assert any("conflict" in w.value.lower() for w in at.warning)
    assert any(e.label.startswith("[1]") for e in at.expander)


def test_demo_unanswerable_question_says_not_found():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Demo (mock data)").run()
    at.text_input(key="question").set_value("Who is the CEO?").run()
    next(b for b in at.button if b.label == "Ask").click().run()
    assert "Not found in the documents" in " ".join(m.value for m in at.markdown)


class FakeRetriever:
    def build(self, chunks):
        self.chunks = chunks

    def retrieve(self, question, top_k=5, doc_ids=None):
        return mocks.mock_passages()[:top_k]


def test_live_pipeline_with_stand_ins(monkeypatch):
    """Real QAPipeline + real reasoning + UI stand-in generator; only chunking/retrieval faked."""
    from src.ingestion import chunking
    from src.pipeline import QAPipeline

    monkeypatch.setattr(chunking, "chunk_document", lambda doc, *a, **k: mocks.mock_chunks())
    pipeline = QAPipeline(Settings(), retriever=FakeRetriever(), generator=services.StandInGenerator())
    result = pipeline._index([mocks.mock_parsed_document()], [])
    assert result.documents_ingested == 1

    resp = pipeline.ask(QueryRequest(question="How many API requests can I make per minute?", top_k=3))
    assert resp.status in {AnswerStatus.CONFLICT, AnswerStatus.ANSWERED}
    assert resp.citations and all(c.excerpt for c in resp.citations)
