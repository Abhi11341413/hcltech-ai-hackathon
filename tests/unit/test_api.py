"""Member 2 - API tests. The pipeline is replaced with a fake so these run without models."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)
todo = pytest.mark.xfail(reason="Member 2: endpoint not implemented yet", strict=False)


def test_health():
    assert client.get("/health").status_code == 200


@todo
def test_ask_validates_input():
    assert client.post("/ask", json={"question": ""}).status_code == 422


@todo
def test_upload_rejects_unsupported_extension():
    r = client.post("/documents", files={"files": ("evil.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 415
    assert r.json()["error_code"] == "UNSUPPORTED_FORMAT"
