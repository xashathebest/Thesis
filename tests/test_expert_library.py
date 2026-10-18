import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.expert_library import router

TOKEN = "synthetic-test-token-0123456789-abcdef"


@pytest.fixture
def library(tmp_path, monkeypatch):
    media = tmp_path / "private_media"; media.mkdir()
    manifest = tmp_path / "private_manifest.json"
    clip = {"id": "dummy", "rule_id": "R0", "title": "DUMMY TEST", "category": "body_weighting",
        "research_question": "Synthetic only", "consultation_date": "2000-01-01", "original_filename": "dummy-original.mp4",
        "original_start_seconds": 10, "original_end_seconds": 20, "clipped_filename": "dummy.mp4", "clipped_length_seconds": 10,
        "claim_kind": "paraphrase", "claim_text": "Synthetic only", "researcher_interpretation": "Test only",
        "validation_status": "proposed", "reviewer": "dummy"}
    manifest.write_text(json.dumps({"schema_version": "expert-library-v1", "clips": [clip]}), encoding="utf-8")
    for key, value in {"EXPERT_LIBRARY_ENABLED": "1", "EXPERT_LIBRARY_SCOPE_VERIFIED": "1",
                       "EXPERT_LIBRARY_RESEARCH_TOKEN": TOKEN, "EXPERT_LIBRARY_MANIFEST": str(manifest),
                       "EXPERT_LIBRARY_MEDIA_ROOT": str(media)}.items():
        monkeypatch.setenv(key, value)
    app = FastAPI(); app.include_router(router)
    client = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 51234))
    return client, manifest, media, clip


def auth():
    return {"Authorization": f"Bearer {TOKEN}"}


def test_default_off_and_unauthorized_are_private(library, monkeypatch):
    client, _, _, _ = library
    assert client.get("/api/expert-library/clips").status_code == 404
    assert client.get("/api/expert-library/clips", headers={"Authorization": "Bearer bad"}).status_code == 404
    monkeypatch.setenv("EXPERT_LIBRARY_SCOPE_VERIFIED", "0")
    assert client.get("/api/expert-library/clips", headers=auth()).status_code == 404
    monkeypatch.setenv("EXPERT_LIBRARY_SCOPE_VERIFIED", "1")
    monkeypatch.setenv("EXPERT_LIBRARY_ENABLED", "0")
    assert client.get("/api/expert-library/clips", headers=auth()).status_code == 404


def test_list_detail_missing_media_and_authenticated_download(library):
    client, _, media, _ = library
    listing = client.get("/api/expert-library/clips", headers=auth())
    assert listing.status_code == 200 and listing.headers["cache-control"] == "no-store"
    assert listing.json()["clips"][0]["id"] == "dummy"
    detail = client.get("/api/expert-library/clips/dummy", headers=auth()).json()
    assert detail["original_start_seconds"] == 10 and detail["original_end_seconds"] == 20
    assert detail["claim_kind"] == "paraphrase" and detail["media_available"] is False
    assert client.get("/api/expert-library/clips/dummy/media", headers=auth()).status_code == 404
    (media / "dummy.mp4").write_bytes(b"DUMMY NOT A REAL RECORDING")
    assert client.get("/api/expert-library/clips/dummy/media").status_code == 404
    response = client.get("/api/expert-library/clips/dummy/media", headers=auth())
    assert response.status_code == 200 and response.content == b"DUMMY NOT A REAL RECORDING"
    assert response.headers["cache-control"] == "no-store"


def test_bad_manifest_and_paths_fail_closed(library):
    client, manifest, _, clip = library
    clip["clipped_filename"] = "../other.mp4"
    manifest.write_text(json.dumps({"schema_version": "expert-library-v1", "clips": [clip]}), encoding="utf-8")
    response = client.get("/api/expert-library/clips", headers=auth())
    assert response.status_code == 503 and "other.mp4" not in response.text
    clip["clipped_filename"] = "dummy.mp4"; clip["original_end_seconds"] = 5
    manifest.write_text(json.dumps({"schema_version": "expert-library-v1", "clips": [clip]}), encoding="utf-8")
    assert client.get("/api/expert-library/clips", headers=auth()).status_code == 503


def test_remote_proxy_and_nonlocal_host_are_denied(library):
    client, _, _, _ = library
    assert client.get("/api/expert-library/clips", headers={**auth(), "X-Forwarded-For": "127.0.0.1"}).status_code == 404
    assert client.get("/api/expert-library/clips", headers={**auth(), "Host": "example.com"}).status_code == 404
    app = FastAPI(); app.include_router(router)
    remote = TestClient(app, base_url="http://localhost", client=("192.0.2.4", 1234))
    assert remote.get("/api/expert-library/clips", headers=auth()).status_code == 404


def test_empty_manifest_has_no_fabricated_examples(library):
    client, manifest, _, _ = library
    manifest.write_text('{"schema_version":"expert-library-v1","clips":[]}', encoding="utf-8")
    assert client.get("/api/expert-library/clips", headers=auth()).json() == {"clips": []}
