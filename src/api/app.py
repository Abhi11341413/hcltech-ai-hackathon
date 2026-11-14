"""FastAPI service (Expected submission: "web interface or API"). OWNER: Member 2.

Endpoints (contract):

    GET  /health               -> {"status": "ok", "documents": int, "chunks": int}
    POST /documents            multipart files[] -> IngestResult   (upload + index)
    POST /ingest/sample        -> IngestResult                     (index data/sample_docs)
    POST /ask                  QueryRequest -> QAResponse
    GET  /documents            -> list of {doc_id, title, source, version}

Errors: catch ``AssistantError`` and return ``ErrorResponse`` with ``exc.http_status``.
Uploads: save to data/uploads/ (git-ignored), validate extension + size (<= 20 MB),
sanitise file names (no path traversal).

The app delegates ALL logic to ``src.pipeline.QAPipeline`` (Member 1).
Run: ``uvicorn src.api.app:app --reload`` -> Swagger UI at http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(
    title="Enterprise Documentation Q&A Assistant",
    version="0.1.0",
    description="Grounded answers with citations over supplied PDF/HTML/Markdown/TXT documents.",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. Member 2: extend with document/chunk counts from the pipeline."""
    return {"status": "ok"}


# Member 2: implement POST /documents, POST /ingest/sample, POST /ask, GET /documents
# and a global exception handler for AssistantError.
