"""FastAPI service (Expected submission: "web interface or API"). OWNER: Member 2.

Endpoints:

    GET    /health                      -> {"status", "documents", "chunks"}
    GET    /documents                   -> list[DocumentInfo]  (title, version, effective date, chunks ...)
    POST   /documents                   multipart files[] -> IngestResult   (upload + vectorise)
    DELETE /documents/{doc_id}          -> RemoveResult   (remove one document from the index)
    DELETE /documents?effective_before=YYYY-MM-DD[&dry_run=true]
                                        -> RemoveResult   (remove stale documents in bulk)
    POST   /ingest/sample               -> IngestResult   (index data/sample_docs)
    POST   /ask                         QueryRequest -> QAResponse

Document lifecycle:

* Uploads are validated (extension, size <= 20 MB, non-empty), file names are sanitised
  (no path traversal) and saved under ``data/uploads/`` (git-ignored), then ingested.
* ``doc_id`` comes from the file name, so uploading a file with the SAME name again
  REPLACES the old version (old chunks are dropped, new ones embedded).
* Deleting a document removes its chunks from the index and, if it was uploaded, its file
  in ``data/uploads/``. Committed sample documents are only un-indexed, never deleted.
* ``data/uploads/`` is the persistent record: on start-up every file there is re-indexed.

Errors: every ``AssistantError`` becomes an ``ErrorResponse`` with ``exc.http_status``.
The app delegates retrieval / answering to ``src.pipeline.QAPipeline`` (Member 1).

Run: ``uvicorn src.api.app:app --reload`` -> Swagger UI at http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import PROJECT_ROOT, get_logger, get_settings
from src.errors import AssistantError, IngestionError, ValidationFailedError
from src.ingestion.loaders import detect_file_type, make_doc_id
from src.pipeline import QAPipeline
from src.schemas import ErrorResponse, FileType, IngestFailure, IngestResult, QAResponse, QueryRequest

logger = get_logger(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_READ_BLOCK = 1024 * 1024


# ---------------------------------------------------------------------------
# API-local errors and response models (not shared contracts)
# ---------------------------------------------------------------------------


class FileTooLargeError(IngestionError):
    error_code = "FILE_TOO_LARGE"
    http_status = 413


class DocumentNotFoundError(AssistantError):
    error_code = "DOCUMENT_NOT_FOUND"
    http_status = 404


class DocumentInfo(BaseModel):
    doc_id: str
    title: str
    source: str
    file_type: FileType
    version: str | None = None
    effective_date: date | None = None
    ingested_at: datetime
    chunks: int
    uploaded: bool = Field(description="True if the file lives in data/uploads/ (deleting it removes the file).")


class RemoveResult(BaseModel):
    removed: list[str] = Field(description="doc_ids removed (or that WOULD be removed when dry_run=true).")
    files_deleted: list[str] = Field(default_factory=list)
    dry_run: bool = False
    documents_remaining: int
    chunks_remaining: int


# ---------------------------------------------------------------------------
# Service state: one pipeline per process, created lazily (so importing the app or
# calling /health never loads the embedding model), guarded by one lock because
# FastAPI runs sync endpoints in a thread pool and index rebuilds are not thread-safe.
# ---------------------------------------------------------------------------


@dataclass
class ApiConfig:
    uploads_dir: Path
    sample_dir: Path
    max_upload_bytes: int = MAX_UPLOAD_BYTES


class ServiceState:
    def __init__(self, config: ApiConfig, pipeline: QAPipeline | None = None) -> None:
        self.config = config
        self.lock = threading.RLock()
        self._pipeline = pipeline

    @property
    def started(self) -> bool:
        return self._pipeline is not None

    def pipeline(self) -> QAPipeline:
        """Create the pipeline on first use and re-index files persisted in data/uploads/."""
        with self.lock:
            if self._pipeline is None:
                self._pipeline = QAPipeline(get_settings())
                self.config.uploads_dir.mkdir(parents=True, exist_ok=True)
                if _upload_files(self.config.uploads_dir):
                    result = self._pipeline.ingest_directory(self.config.uploads_dir)
                    logger.info("restored %d uploaded documents", result.documents_ingested)
            return self._pipeline


def _default_state() -> ServiceState:
    settings = get_settings()
    return ServiceState(ApiConfig(uploads_dir=PROJECT_ROOT / "data" / "uploads",
                                  sample_dir=settings.resolve(settings.data_dir)))


app = FastAPI(
    title="Enterprise Documentation Q&A Assistant",
    version="0.2.0",
    description="Grounded answers with citations over supplied PDF/HTML/Markdown/TXT documents.",
)
app.state.service = _default_state()


def get_service() -> ServiceState:
    """FastAPI dependency; tests replace ``app.state.service`` with one holding a fake pipeline."""
    return app.state.service


Service = Annotated[ServiceState, Depends(get_service)]


@app.exception_handler(AssistantError)
def _assistant_error_handler(_: Request, exc: AssistantError) -> JSONResponse:
    body = ErrorResponse(error_code=exc.error_code, message=exc.message, details=exc.details or None)
    return JSONResponse(status_code=exc.http_status, content=body.model_dump())


@app.exception_handler(NotImplementedError)
def _not_implemented_handler(_: Request, exc: NotImplementedError) -> JSONResponse:
    """A teammate's module is not merged yet (e.g. generation): say so instead of a bare 500."""
    body = ErrorResponse(error_code="NOT_IMPLEMENTED", message=str(exc) or "not implemented yet")
    return JSONResponse(status_code=501, content=body.model_dump())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _upload_files(directory: Path) -> list[Path]:
    return [p for p in directory.iterdir() if p.is_file() and not p.name.startswith(".")] \
        if directory.is_dir() else []


def sanitize_filename(name: str | None) -> str:
    """Keep only the base name, replace unsafe characters, refuse empty/hidden names.

    'C:\\x\\..\\Billing Policy (v2).md' -> 'Billing_Policy_v2_.md'

    Raises:
        ValidationFailedError: nothing usable is left.
    """
    base = re.split(r"[\\/]", name or "")[-1]
    stem, dot, ext = base.rpartition(".")
    if not dot:
        stem, ext = base, ""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("._")
    ext = re.sub(r"[^A-Za-z0-9]", "", ext)
    if not stem:
        raise ValidationFailedError("invalid file name", details={"filename": (name or "")[:100]})
    return f"{stem[:100]}.{ext}" if ext else stem[:100]


def sanitize_or_raw(name: str | None) -> str:
    """Sanitised name for failure reports, or the truncated raw name if it cannot be sanitised."""
    try:
        return sanitize_filename(name)
    except ValidationFailedError:
        return (name or "")[:100]


def _save_upload(upload: UploadFile, config: ApiConfig) -> Path:
    """Validate and store one upload; returns the saved path.

    Raises:
        ValidationFailedError, UnsupportedFormatError, FileTooLargeError, IngestionError
    """
    safe_name = sanitize_filename(upload.filename)
    detect_file_type(Path(safe_name))  # raises UnsupportedFormatError before anything is written
    config.uploads_dir.mkdir(parents=True, exist_ok=True)
    target = config.uploads_dir / safe_name
    partial = config.uploads_dir / f".{safe_name}.part"
    size = 0
    try:
        with partial.open("wb") as out:
            while block := upload.file.read(_READ_BLOCK):
                size += len(block)
                if size > config.max_upload_bytes:
                    raise FileTooLargeError(f"{safe_name} exceeds {config.max_upload_bytes // (1024 * 1024)} MB",
                                            details={"source": safe_name})
                out.write(block)
        if size == 0:
            raise IngestionError(f"{safe_name} is empty", details={"source": safe_name})
        partial.replace(target)  # atomic: a re-upload replaces the old file only when complete
    finally:
        partial.unlink(missing_ok=True)
    return target


def _uploaded_file(doc_id: str, source: str, config: ApiConfig) -> Path | None:
    """The file in the uploads dir that backs ``doc_id``, if any.

    Matched by file name + doc_id (not by resolving ``source``), so it works wherever the
    uploads dir lives. A document whose upload was later superseded by a same-id sample file
    still owns that upload, so deleting the document also stops it coming back on restart.
    """
    path = config.uploads_dir / Path(source.replace("\\", "/")).name
    return path if path.is_file() and make_doc_id(path) == doc_id else None


def _remove(service: ServiceState, doc_ids: list[str], dry_run: bool = False) -> RemoveResult:
    pipeline = service.pipeline()
    with service.lock:
        by_id = {d.doc_id: d for d in pipeline.documents}
        targets = [d for d in dict.fromkeys(doc_ids) if d in by_id]
        files_deleted: list[str] = []
        if not dry_run and targets:
            pipeline.remove_documents(targets)
            for doc_id in targets:
                path = _uploaded_file(doc_id, by_id[doc_id].source, service.config)
                if path is not None:
                    path.unlink()
                    files_deleted.append(path.name)
        return RemoveResult(removed=targets, files_deleted=files_deleted, dry_run=dry_run,
                            documents_remaining=len(pipeline.documents),
                            chunks_remaining=pipeline.num_chunks)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health(service: Service) -> dict[str, str | int]:
    """Liveness probe with corpus size. Does not load models if nothing was ingested yet."""
    if not service.started:
        return {"status": "ok", "documents": 0, "chunks": 0}
    pipeline = service.pipeline()
    return {"status": "ok", "documents": len(pipeline.documents), "chunks": pipeline.num_chunks}


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents(service: Service) -> list[DocumentInfo]:
    """Indexed documents, oldest effective date first (undated last), to spot stale ones."""
    pipeline = service.pipeline()
    with service.lock:
        counts = {d: 0 for d in (doc.doc_id for doc in pipeline.documents)}
        for chunk in pipeline.all_chunks():
            counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
        docs = sorted(pipeline.documents, key=lambda d: (d.effective_date is None, d.effective_date or date.max))
        return [DocumentInfo(doc_id=d.doc_id, title=d.title, source=d.source, file_type=d.file_type,
                             version=d.version, effective_date=d.effective_date, ingested_at=d.ingested_at,
                             chunks=counts.get(d.doc_id, 0), uploaded=_uploaded_file(d.doc_id, d.source, service.config) is not None)
                for d in docs]


@app.post("/documents", response_model=IngestResult)
def upload_documents(files: Annotated[list[UploadFile], File(description="PDF, HTML, Markdown or TXT files")],
                     service: Service) -> IngestResult:
    """Upload and index documents. A file with the same name as an indexed document replaces it.

    Invalid files are reported in ``failures`` while valid ones are still indexed. If NO file
    is valid, the first error is returned with its HTTP status (415 unsupported, 413 too large).
    """
    saved: list[Path] = []
    failures: list[IngestFailure] = []
    first_error: AssistantError | None = None
    for upload in files:
        try:
            saved.append(_save_upload(upload, service.config))
        except AssistantError as exc:
            first_error = first_error or exc
            failures.append(IngestFailure(source=sanitize_or_raw(upload.filename), error_code=exc.error_code,
                                          message=exc.message))
    if not saved:
        raise first_error or ValidationFailedError("no files uploaded")

    pipeline = service.pipeline()
    with service.lock:
        result = pipeline.ingest_paths(saved)
    failed_sources = {f.source for f in result.failures}
    for path in saved:  # do not keep files that could not be parsed: they would fail again on restart
        if str(path) in failed_sources:
            path.unlink(missing_ok=True)
    return result.model_copy(update={"failures": failures + result.failures})



@app.delete("/documents/{doc_id}", response_model=RemoveResult)
def delete_document(doc_id: str, service: Service) -> RemoveResult:
    """Remove one document (e.g. an outdated version) from the index.

    Raises:
        DocumentNotFoundError (404): no indexed document has this id.
    """
    result = _remove(service, [doc_id])
    if not result.removed:
        raise DocumentNotFoundError(f"No indexed document with id '{doc_id}'", details={"doc_id": doc_id})
    return result


@app.delete("/documents", response_model=RemoveResult)
def delete_stale_documents(
    effective_before: Annotated[date, Query(description="Remove documents whose effective date is before "
                                                       "this date. Documents without a date are kept.")],
    service: Service,
    dry_run: Annotated[bool, Query(description="Only list what would be removed.")] = False,
) -> RemoveResult:
    """Remove stale documents in bulk by effective date (use dry_run=true first)."""
    pipeline = service.pipeline()
    stale = [d.doc_id for d in pipeline.documents if d.effective_date and d.effective_date < effective_before]
    return _remove(service, stale, dry_run=dry_run)


@app.post("/ingest/sample", response_model=IngestResult)
def ingest_sample(service: Service) -> IngestResult:
    """Index the committed synthetic corpus in data/sample_docs/ (replaces same-id documents)."""
    pipeline = service.pipeline()
    with service.lock:
        return pipeline.ingest_directory(service.config.sample_dir)


@app.post("/ask", response_model=QAResponse)
def ask(request: QueryRequest, service: Service) -> QAResponse:
    """Answer a question from the indexed documents (409 INDEX_NOT_BUILT if nothing is indexed)."""
    pipeline = service.pipeline()
    with service.lock:
        return pipeline.ask(request)

