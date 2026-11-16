"""Streamlit web interface for the Enterprise Documentation Q&A Assistant.

Run from the repo root:
    streamlit run src/ui/app.py

The page is a thin view over ``src/ui/services.py`` and ``src/pipeline.QAPipeline``.
It works at every stage of the hackathon: modules that are not merged yet are shown
in the sidebar, and Demo mode shows the interface with mock data.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run` puts this file's folder on sys.path, not the repo root.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.errors import AssistantError  # noqa: E402
from src.schemas import IngestResult, QAResponse, QueryRequest  # noqa: E402
from src.ui import services  # noqa: E402

st.set_page_config(page_title="Enterprise Docs Q&A", layout="wide")


# --------------------------------------------------------------------------- state
@st.cache_resource(show_spinner=False)
def _readiness() -> dict[str, services.ModuleStatus]:
    return services.check_readiness()


def _state():
    ss = st.session_state
    ss.setdefault("bundle", None)          # PipelineBundle for the current config
    ss.setdefault("bundle_key", None)
    ss.setdefault("question", "")
    ss.setdefault("response", None)
    ss.setdefault("last_ingest", None)
    ss.setdefault("eval_report", None)
    return ss


def _get_bundle(ss, statuses, offline: bool) -> services.PipelineBundle | None:
    """Build (or reuse) the live pipeline for the current settings."""
    key = ("live", offline)
    if ss.bundle is not None and ss.bundle_key == key:
        return ss.bundle
    settings, _ = services.effective_settings(get_settings(), prefer_offline=offline)
    try:
        ss.bundle = services.build_pipeline(settings, statuses)
        ss.bundle_key = key
        ss.response = None
        ss.last_ingest = None
        ss.eval_report = None
    except AssistantError as exc:
        st.error(exc.message)
        return None
    return ss.bundle


# --------------------------------------------------------------------------- sidebar
def sidebar(statuses) -> tuple[str, bool]:
    st.sidebar.title("Enterprise Docs Q&A")
    st.sidebar.caption("Grounded answers with citations, from your documents only.")

    st.sidebar.subheader("Team modules")
    for status in statuses.values():
        icon = ":green[●]" if status.ready else ":orange[○]"
        text = "ready" if status.ready else "not merged yet"
        st.sidebar.markdown(f"{icon} **{status.label}** · {status.owner} · {text}")
    if st.sidebar.button("Re-check modules", use_container_width=True):
        _readiness.clear()
        st.session_state.bundle = None
        st.rerun()

    live_ok = services.live_mode_available(statuses)
    st.sidebar.subheader("Mode")
    options = ["Live pipeline", "Demo (mock data)"] if live_ok else ["Demo (mock data)"]
    mode = st.sidebar.radio("Mode", options, label_visibility="collapsed")
    if not live_ok:
        st.sidebar.info("Live mode turns on automatically once ingestion, retrieval and reasoning are merged.")

    offline = False
    if mode == "Live pipeline":
        offline = st.sidebar.toggle("Offline (no LLM calls)", value=not services.has_api_key(),
                                    disabled=not services.has_api_key(),
                                    help="Without an LLM key in .env (ANTHROPIC_API_KEY, GEMINI_API_KEY or "
                                         "SARVAM_API_KEY) the app always runs offline.")
        _, explanation = services.effective_settings(get_settings(), prefer_offline=offline)
        st.sidebar.caption(explanation)
    return mode, offline


# --------------------------------------------------------------------------- rendering
def render_response(resp: QAResponse, demo: bool) -> None:
    label, color = services.STATUS_STYLE[resp.status]
    st.markdown(f"#### :{color}[● {label}]")
    if demo:
        st.caption("Demo data from src/mocks.py. Switch to Live mode once all modules are merged.")

    st.markdown(resp.answer)

    c1, c2 = st.columns([3, 1])
    with c1:
        st.progress(resp.confidence.score,
                    text=f"Confidence: {resp.confidence.level.value} ({resp.confidence.score:.2f})")
        if resp.confidence.reasons:
            st.caption("Why: " + "; ".join(resp.confidence.reasons))
    with c2:
        if resp.latency_ms is not None:
            st.metric("Latency", f"{resp.latency_ms:.0f} ms")

    if resp.conflicts:
        st.markdown("##### Conflicting sources")
        titles = {p.chunk.chunk_id: f"{p.chunk.doc_title} · {p.chunk.section}" for p in resp.retrieved}
        titles.update({c.chunk_id: f"{c.doc_title} · {c.section}" for c in resp.citations})
        for conflict in resp.conflicts:
            lines = [f"**{conflict.conflict_type.value.title()} conflict:** {conflict.description}"]
            lines += [f"- *{titles.get(cid, cid)}*: {value}" for cid, value in conflict.values.items()]
            st.warning("\n".join(lines))

    if resp.citations:
        st.markdown("##### Citations")
        for cit in resp.citations:
            where = cit.section + (f" · page {cit.page}" if cit.page else "")
            with st.expander(f"[{cit.citation_id}] {cit.doc_title} · {where}"):
                st.markdown(f"> {cit.excerpt}")
                st.caption(f"Source: {cit.source} · chunk {cit.chunk_id}")

    if resp.retrieved:
        with st.expander(f"Retrieved passages ({len(resp.retrieved)})"):
            st.dataframe(
                [{"rank": p.rank, "score": p.score, "document": p.chunk.doc_title, "section": p.chunk.section,
                  "version": p.chunk.version or "", "signals": ", ".join(f"{k}={v}" for k, v in
                                                                         p.score_breakdown.items()),
                  "text": p.chunk.text[:160]} for p in resp.retrieved],
                hide_index=True, use_container_width=True)


def render_ingest(result: IngestResult) -> None:
    st.success(f"Indexed {result.documents_ingested} document(s), {result.chunks_created} passage(s).")
    for failure in result.failures:
        st.error(f"{Path(failure.source).name}: {failure.error_code} · {failure.message}")


# --------------------------------------------------------------------------- tabs
def ask_tab(ss, mode: str, bundle) -> None:
    st.subheader("Ask a question")
    cols = st.columns(len(services.EXAMPLE_QUESTIONS))
    for col, (kind, question) in zip(cols, services.EXAMPLE_QUESTIONS.items(), strict=True):
        if col.button(kind, use_container_width=True, help=question):
            ss.question = question

    question = st.text_input("Question", key="question", placeholder="e.g. How do I configure single sign-on?")
    with st.expander("Options"):
        top_k = st.slider("Passages to retrieve (top-k)", 1, 10, 5)
        doc_ids, version = None, None
        if bundle is not None and bundle.pipeline.documents:
            docs = {d.doc_id: d.title for d in bundle.pipeline.documents}
            chosen = st.multiselect("Only search these documents", list(docs), format_func=docs.get)
            doc_ids = chosen or None
        version = st.text_input("Product version (optional)", placeholder="e.g. 3.0") or None

    if st.button("Ask", type="primary", disabled=not question.strip()):
        try:
            request = QueryRequest(question=question, top_k=top_k, doc_ids=doc_ids, version=version)
        except ValueError as exc:
            st.error(f"Invalid question: {exc}")
            return
        if mode.startswith("Demo"):
            ss.response = (services.demo_response(request.question), True)
        elif bundle is not None:
            with st.spinner("Retrieving and answering..."):
                try:
                    ss.response = (bundle.pipeline.ask(request), False)
                except NotImplementedError as exc:
                    st.error(f"This step needs {services.missing_owner(exc)}'s module, which is not merged yet.")
                except AssistantError as exc:
                    st.error(exc.message)

    if ss.response is not None:
        st.divider()
        render_response(*ss.response)


def documents_tab(ss, mode: str, bundle) -> None:
    st.subheader("Knowledge base")
    if mode.startswith("Demo") or bundle is None:
        st.info("Document upload works in Live mode, which needs the ingestion, retrieval and reasoning modules.")
        return
    for note in bundle.notes:
        st.caption(note)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sample corpus** (13 synthetic documents)")
        if st.button("Load sample documents", use_container_width=True):
            with st.spinner("Loading, chunking and indexing (first run downloads the embedding model)..."):
                try:
                    ss.last_ingest = bundle.pipeline.ingest_directory(services.SAMPLE_DOCS_DIR)
                except NotImplementedError as exc:
                    st.error(f"Ingestion needs {services.missing_owner(exc)}'s module.")
                except AssistantError as exc:
                    st.error(exc.message)
    with c2:
        settings = get_settings()
        files = st.file_uploader("Upload PDF, HTML, Markdown or TXT", accept_multiple_files=True,
                                 type=[e.lstrip(".") for e in settings.ingestion.supported_extensions])
        if files and st.button("Index uploaded files", use_container_width=True):
            paths = []
            for f in files:
                try:
                    paths.append(services.save_upload(f.name, f.getvalue(), settings))
                except AssistantError as exc:
                    st.error(exc.message)
            if paths:
                with st.spinner("Indexing uploads..."):
                    ss.last_ingest = bundle.pipeline.ingest_paths(paths)

    if ss.last_ingest is not None:
        render_ingest(ss.last_ingest)

    docs = bundle.pipeline.documents
    st.markdown(f"**Indexed:** {len(docs)} document(s), {bundle.pipeline.num_chunks} passage(s)")
    if docs:
        st.dataframe([{"doc_id": d.doc_id, "title": d.title, "format": d.file_type.value,
                       "version": d.version or "", "effective": str(d.effective_date or ""),
                       "sections": len(d.sections)} for d in docs], hide_index=True, use_container_width=True)


def evaluation_tab(ss, mode: str, bundle, statuses) -> None:
    st.subheader("Evaluation")
    st.caption("Runs the labelled question set (data/eval/questions.jsonl) through the live pipeline.")
    if mode.startswith("Demo") or bundle is None or not statuses["evaluation"].ready:
        st.info("Available in Live mode once the evaluation module is merged.")
        return
    if not bundle.pipeline.documents:
        st.info("Load the sample documents in the Knowledge base tab first.")
        return
    settings = get_settings()
    if bundle.uses_llm:
        st.warning("Live LLM mode: this sends one request per question (about 27 calls).")
    if st.button("Run evaluation", type="primary"):
        from src.evaluation.run_eval import load_questions, run_evaluation
        with st.spinner("Asking every labelled question..."):
            try:
                questions = load_questions(settings.resolve(settings.evaluation.questions_path))
                ss.eval_report = run_evaluation(bundle.pipeline, questions, k=settings.evaluation.k)
            except AssistantError as exc:
                st.error(exc.message)

    report = ss.eval_report
    if report is None:
        return
    st.markdown(f"**{report.n_questions} questions, k = {report.k}**")
    items = list(report.metrics.items())
    for start in range(0, len(items), 4):
        cols = st.columns(4)
        for col, (name, value) in zip(cols, items[start:start + 4], strict=False):
            col.metric(name, f"{value:.3f}" if isinstance(value, float) else value)
    if report.per_category:
        st.markdown("##### By question category")
        st.dataframe([{"category": cat, **{k: round(v, 3) for k, v in vals.items()}}
                      for cat, vals in report.per_category.items()], hide_index=True, use_container_width=True)
    st.markdown("##### Per question")
    st.dataframe([{"qid": r.qid, "category": r.category.value, "expected": r.expected_status.value,
                   "predicted": r.predicted_status.value, "correct": r.expected_status == r.predicted_status,
                   "recall@k": r.recall_at_k, "MRR": r.reciprocal_rank, "groundedness": r.groundedness,
                   "error": r.error or ""} for r in report.results], hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------- main
def main() -> None:
    ss = _state()
    statuses = _readiness()
    mode, offline = sidebar(statuses)
    bundle = _get_bundle(ss, statuses, offline) if mode == "Live pipeline" else None
    if bundle is not None:
        st.sidebar.caption(f"Answer generator: {bundle.generator_label}")

    st.title("Enterprise Documentation Q&A Assistant")
    if mode == "Live pipeline" and bundle is not None and not bundle.pipeline.documents:
        st.info("Start in the **Knowledge base** tab: load the sample documents or upload your own.")

    ask, docs, evaluation = st.tabs(["Ask", "Knowledge base", "Evaluation"])
    with ask:
        ask_tab(ss, mode, bundle)
    with docs:
        documents_tab(ss, mode, bundle)
    with evaluation:
        evaluation_tab(ss, mode, bundle, statuses)


main()
