"""UI service layer tests (no Streamlit needed)."""

from __future__ import annotations

import pytest

from src import mocks
from src.config import Settings
from src.errors import UnsupportedFormatError, ValidationFailedError
from src.schemas import AnswerStatus
from src.ui import services


def _statuses(**ready: bool) -> dict[str, services.ModuleStatus]:
    keys = ["ingestion", "retrieval", "generation", "reasoning", "evaluation"]
    return {k: services.ModuleStatus(key=k, label=k, owner="Member X", ready=ready.get(k, True)) for k in keys}


def test_probe_distinguishes_placeholders_from_implemented_code():
    def placeholder():
        raise NotImplementedError("Member 4: implement x")

    def implemented_but_rejects_input():
        raise ValueError("bad probe input")

    assert services._probe([placeholder]) == (False, "Member 4: implement x")
    assert services._probe([implemented_but_rejects_input, lambda: 1]) == (True, "")


def test_check_readiness_reports_every_module():
    statuses = services.check_readiness()
    assert set(statuses) == {"ingestion", "retrieval", "generation", "reasoning", "evaluation"}
    assert all(s.owner.startswith("Member") for s in statuses.values())


def test_missing_owner():
    assert services.missing_owner(NotImplementedError("Member 2: implement load_pdf")) == "Member 2"
    assert services.missing_owner(NotImplementedError("todo")) == "a team member"


def test_live_mode_needs_ingestion_retrieval_reasoning():
    assert services.live_mode_available(_statuses())
    assert services.live_mode_available(_statuses(generation=False))
    assert not services.live_mode_available(_statuses(ingestion=False))


def test_effective_settings_falls_back_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings, why = services.effective_settings(Settings())
    assert settings.generation.provider == "extractive" and "ANTHROPIC_API_KEY" in why
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-real-key")
    settings, _ = services.effective_settings(Settings())
    assert settings.generation.provider == "anthropic"
    settings, _ = services.effective_settings(Settings(), prefer_offline=True)
    assert settings.generation.provider == "extractive"


def test_effective_settings_does_not_mutate_base(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    base = Settings()
    services.effective_settings(base)
    assert base.generation.provider == "anthropic"


def test_stand_in_generator_cites_verbatim_excerpts():
    passages = mocks.mock_passages()
    answer = services.StandInGenerator().generate("How many requests per minute?", passages)
    assert answer.model == services.STAND_IN_MODEL and answer.groundedness_score == 1.0
    assert [c.citation_id for c in answer.citations] == [1, 2]
    for cit, passage in zip(answer.citations, passages, strict=False):
        assert cit.excerpt in passage.chunk.text
    assert "[1]" in answer.answer_text and "[2]" in answer.answer_text


def test_stand_in_generator_without_passages_is_insufficient():
    assert services.StandInGenerator().generate("q?", []).insufficient_evidence


def test_build_pipeline_refuses_when_core_modules_missing():
    with pytest.raises(ValidationFailedError, match="ingestion"):
        services.build_pipeline(Settings(), _statuses(ingestion=False))


def test_build_pipeline_stands_in_for_generation():
    bundle = services.build_pipeline(Settings(), _statuses(generation=False))
    assert isinstance(bundle.pipeline.generator, services.StandInGenerator)
    assert "stand-in" in bundle.generator_label and bundle.notes


def test_sanitize_filename_blocks_path_traversal():
    assert services.sanitize_filename("../../etc/evil name.md") == "evil_name.md"
    assert services.sanitize_filename("C:\\Users\\x\\doc.pdf") == "doc.pdf"
    with pytest.raises(ValidationFailedError):
        services.sanitize_filename("../..")


def test_save_upload_validates_type_size_and_writes(tmp_path):
    settings = Settings()
    path = services.save_upload("guide.md", b"# Title\ntext", settings, upload_dir=tmp_path)
    assert path.read_bytes() == b"# Title\ntext" and path.parent == tmp_path
    with pytest.raises(UnsupportedFormatError):
        services.save_upload("virus.exe", b"MZ", settings, upload_dir=tmp_path)
    with pytest.raises(ValidationFailedError):
        services.save_upload("empty.txt", b"", settings, upload_dir=tmp_path)
    with pytest.raises(ValidationFailedError):
        services.save_upload("big.txt", b"x" * (services.MAX_UPLOAD_BYTES + 1), settings, upload_dir=tmp_path)


@pytest.mark.parametrize(("kind", "status"), [
    ("Direct", AnswerStatus.ANSWERED),
    ("Paraphrase", AnswerStatus.ANSWERED),
    ("Conflict", AnswerStatus.CONFLICT),
    ("Ambiguous", AnswerStatus.NEEDS_CLARIFICATION),
    ("Unanswerable", AnswerStatus.NOT_FOUND),
])
def test_demo_responses_cover_every_status(kind, status):
    question = services.EXAMPLE_QUESTIONS[kind]
    resp = services.demo_response(question)
    assert resp.status == status and resp.question == question


def test_status_style_covers_every_status():
    assert set(services.STATUS_STYLE) == set(AnswerStatus)
