"""Member 2 - API tests. The pipeline gets a fake retriever/generator so these run without models.

Real loaders and chunking are used, so uploads are genuinely parsed and indexed.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import ApiConfig, ServiceState, app, sanitize_filename
from src.config import Settings
from src.errors import ValidationFailedError
from src.pipeline import QAPipeline
from src.schemas import Chunk, GeneratedAnswer, RetrievedPassage, RetrieverName

client = TestClient(app)

V1 = b"# Limits Guide (v1.0)\n\nLast updated: 2025-01-10\n\n## Rate limits\n\nThe API allows 100 requests per minute.\n"
V2 = b"# Limits Guide (v2.0)\n\nLast updated: 2026-06-01\n\n## Rate limits\n\nThe API allows 300 requests per minute.\n"


class FakeRetriever:
    """Keyword-overlap retriever: enough to prove which chunks are in the index."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.builds = 0

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self.builds += 1

    def retrieve(self, question: str, top_k: int = 5, doc_ids: list[str] | None = None) -> list[RetrievedPassage]:
        words = set(question.lower().split())
        scored = [(len(words & set(c.text.lower().split())), c) for c in self.chunks
                  if doc_ids is None or c.doc_id in doc_ids]
        scored = sorted((s for s in scored if s[0] > 0), key=lambda s: -s[0])[:top_k]
        return [RetrievedPassage(chunk=c, score=0.9, rank=i, retriever=RetrieverName.HYBRID)
                for i, (_, c) in enumerate(scored, start=1)]


class FakeGenerator:
    def generate(self, question: str, passages: list[RetrievedPassage], version: str | None = None) -> GeneratedAnswer:
        return GeneratedAnswer(answer_text=passages[0].chunk.text if passages else "", model="fake",
                               insufficient_evidence=not passages, groundedness_score=1.0)


@pytest.fixture
def api(tmp_path, sample_docs_dir):
    """Point the app at a temp uploads dir and a fake-backed pipeline; restore afterwards."""
    original = app.state.service
    retriever = FakeRetriever()
    pipeline = QAPipeline(Settings(), retriever=retriever, generator=FakeGenerator())
    app.state.service = ServiceState(ApiConfig(uploads_dir=tmp_path / "uploads", sample_dir=sample_docs_dir,
                                               max_upload_bytes=1024), pipeline=pipeline)
    yield client, tmp_path / "uploads", retriever
    app.state.service = original


def _upload(c: TestClient, name: str, content: bytes, mime: str = "text/markdown"):
    return c.post("/documents", files=[("files", (name, content, mime))])


# ----------------------------------------------------------------------------- contract tests


def test_health():
    assert client.get("/health").status_code == 200


def test_ask_validates_input():
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_upload_rejects_unsupported_extension():
    r = client.post("/documents", files={"files": ("evil.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 415
    assert r.json()["error_code"] == "UNSUPPORTED_FORMAT"


# ----------------------------------------------------------------------------- upload


def test_upload_indexes_document_and_lists_it(api):
    c, uploads, retriever = api
    r = _upload(c, "limits_guide.md", V1)
    assert r.status_code == 200, r.text
    assert r.json()["documents_ingested"] == 1 and r.json()["chunks_created"] >= 1
    assert (uploads / "limits_guide.md").is_file()
    assert {ch.doc_id for ch in retriever.chunks} == {"limits-guide"}

    docs = c.get("/documents").json()
    assert docs[0]["doc_id"] == "limits-guide" and docs[0]["version"] == "1.0"
    assert docs[0]["effective_date"] == "2025-01-10" and docs[0]["uploaded"] is True
    assert c.get("/health").json()["documents"] == 1


def test_reupload_same_name_replaces_old_version(api):
    c, _, retriever = api
    _upload(c, "limits_guide.md", V1)
    _upload(c, "limits_guide.md", V2)
    texts = " ".join(ch.text for ch in retriever.chunks)
    assert "300 requests" in texts and "100 requests" not in texts
    docs = c.get("/documents").json()
    assert len(docs) == 1 and docs[0]["version"] == "2.0"


def test_mixed_batch_reports_bad_files_and_indexes_good_ones(api):
    c, uploads, _ = api
    r = c.post("/documents", files=[("files", ("good.md", V1, "text/markdown")),
                                    ("files", ("bad.docx", b"x", "application/octet-stream")),
                                    ("files", ("huge.txt", b"a" * 2048, "text/plain"))])
    body = r.json()
    assert r.status_code == 200 and body["documents_ingested"] == 1
    assert {f["error_code"] for f in body["failures"]} == {"UNSUPPORTED_FORMAT", "FILE_TOO_LARGE"}
    assert sorted(p.name for p in uploads.iterdir()) == ["good.md"]  # no partial or rejected files left


def test_too_large_single_file_is_413(api):
    c, _, _ = api
    r = _upload(c, "huge.txt", b"a" * 2048, "text/plain")
    assert r.status_code == 413 and r.json()["error_code"] == "FILE_TOO_LARGE"


def test_unparseable_upload_is_reported_and_file_removed(api):
    c, uploads, _ = api
    r = _upload(c, "broken.pdf", b"%PDF-1.4 this is not really a pdf", "application/pdf")
    assert r.status_code == 200 and r.json()["documents_ingested"] == 0 and r.json()["failures"]
    assert not (uploads / "broken.pdf").exists()


def test_path_traversal_names_are_sanitised(api):
    c, uploads, _ = api
    r = _upload(c, "..\\..\\etc/../Billing Policy (v2).md", V1)
    assert r.status_code == 200
    assert [p.name for p in uploads.iterdir()] == ["Billing_Policy_v2.md"]


@pytest.mark.parametrize("name", ["", "...", "/"])
def test_sanitize_rejects_empty_names(name):
    with pytest.raises(ValidationFailedError):
        sanitize_filename(name)


# ----------------------------------------------------------------------------- delete


def test_delete_document_removes_chunks_and_uploaded_file(api):
    c, uploads, retriever = api
    _upload(c, "limits_guide.md", V1)
    _upload(c, "other_doc.md", V2)
    r = c.delete("/documents/limits-guide")
    assert r.status_code == 200, r.text
    assert r.json()["removed"] == ["limits-guide"] and r.json()["files_deleted"] == ["limits_guide.md"]
    assert {ch.doc_id for ch in retriever.chunks} == {"other-doc"}
    assert not (uploads / "limits_guide.md").exists()


def test_delete_unknown_document_is_404(api):
    c, _, _ = api
    r = c.delete("/documents/nope")
    assert r.status_code == 404 and r.json()["error_code"] == "DOCUMENT_NOT_FOUND"


def test_delete_last_document_then_ask_is_409(api):
    c, _, _ = api
    _upload(c, "limits_guide.md", V1)
    assert c.delete("/documents/limits-guide").json()["documents_remaining"] == 0
    r = c.post("/ask", json={"question": "How many requests per minute?"})
    assert r.status_code == 409 and r.json()["error_code"] == "INDEX_NOT_BUILT"


def test_sample_docs_are_unindexed_but_never_deleted_from_disk(api, sample_docs_dir):
    c, _, _ = api
    assert c.post("/ingest/sample").json()["documents_ingested"] == 13
    r = c.delete("/documents/support-faq-v1")
    assert r.json()["removed"] == ["support-faq-v1"] and r.json()["files_deleted"] == []
    assert (sample_docs_dir / "support_faq_v1.txt").is_file()
    assert "support-faq-v1" not in {d["doc_id"] for d in c.get("/documents").json()}


def test_delete_stale_by_effective_date_with_dry_run(api):
    c, _, _ = api
    c.post("/ingest/sample")
    preview = c.delete("/documents", params={"effective_before": "2026-01-01", "dry_run": True}).json()
    assert sorted(preview["removed"]) == ["security-faq-v2", "support-faq-v1"]
    assert preview["dry_run"] is True and preview["documents_remaining"] == 13

    done = c.delete("/documents", params={"effective_before": "2026-01-01"}).json()
    assert sorted(done["removed"]) == ["security-faq-v2", "support-faq-v1"]
    assert done["documents_remaining"] == 11
    undated = {"account-access", "integrations-configuration"}  # no effective date -> kept
    assert undated <= {d["doc_id"] for d in c.get("/documents").json()}


def test_delete_stale_requires_a_date(api):
    c, _, _ = api
    assert c.delete("/documents").status_code == 422


# ----------------------------------------------------------------------------- ask


def test_ask_uses_uploaded_document(api):
    c, _, _ = api
    _upload(c, "limits_guide.md", V2)
    r = c.post("/ask", json={"question": "How many requests per minute does the API allow?"})
    assert r.status_code == 200, r.text
    assert r.json()["retrieved"][0]["chunk"]["doc_id"] == "limits-guide"


def test_restart_restores_uploaded_documents(tmp_path, monkeypatch):
    """A fresh ServiceState re-indexes whatever is in data/uploads/."""
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "limits_guide.md").write_bytes(V2)
    monkeypatch.setattr("src.api.app.QAPipeline",
                        lambda settings: QAPipeline(settings, retriever=FakeRetriever(), generator=FakeGenerator()))
    state = ServiceState(ApiConfig(uploads_dir=uploads, sample_dir=tmp_path))
    assert [d.doc_id for d in state.pipeline().documents] == ["limits-guide"]


def test_unmerged_module_returns_501(api):
    c, _, _ = api

    class NotReady(FakeGenerator):
        def generate(self, *a, **k):
            raise NotImplementedError("Member 4: implement AnswerGenerator.generate")

    app.state.service.pipeline().generator = NotReady()
    _upload(c, "limits_guide.md", V2)
    r = TestClient(app, raise_server_exceptions=False).post(
        "/ask", json={"question": "How many requests per minute does the API allow?"})
    assert r.status_code == 501 and r.json()["error_code"] == "NOT_IMPLEMENTED"
