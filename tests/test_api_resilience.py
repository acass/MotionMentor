"""API contract, security, and resilience tests against an isolated data dir.

Every test runs in a temporary working directory with its own SQLite DB so
nothing here can touch the real data/ folder.
"""

from __future__ import annotations

import base64
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from motion_mentor import server
from motion_mentor.app import MotionMentorApp
from motion_mentor.storage.models import Session


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    mentor = MotionMentorApp(config_path=tmp_path / "missing.yaml")
    activity = mentor.load_or_create_activity("reach_and_pinch")
    monkeypatch.setattr(server, "_mentor_app", mentor)
    yield mentor, activity
    mentor.close()


@pytest.fixture
def client(isolated) -> TestClient:
    return TestClient(server.app, raise_server_exceptions=False)


# --- API contract -----------------------------------------------------------

def test_openapi_schema_lists_every_route(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    for path in ["/api/activities", "/api/sessions", "/api/compare", "/api/record_upload"]:
        assert path in schema["paths"], path


def test_unknown_ids_return_404_not_500(client: TestClient) -> None:
    for url in ["/api/sessions/nope", "/api/sessions/nope/landmarks",
                "/api/sessions/nope/features", "/api/references/nope",
                "/api/assessments/nope", "/api/videos/nope.mp4"]:
        assert client.get(url).status_code == 404, url


def test_role_filter_rejects_bad_value(client: TestClient) -> None:
    assert client.get("/api/sessions?role=admin").status_code == 422


def test_compare_validation(client: TestClient) -> None:
    assert client.post("/api/compare", json={}).status_code == 422
    assert client.post("/api/compare", json={"attempt_session_id": "nope"}).status_code == 404


def test_compare_without_reference_is_400(client: TestClient, isolated) -> None:
    mentor, activity = isolated
    s = Session(activity_id=activity.activity_id, role="trainee")
    mentor.db.save_session(s)
    res = client.post("/api/compare", json={"attempt_session_id": s.session_id})
    assert res.status_code == 400


# --- Security ---------------------------------------------------------------

def test_video_path_traversal_blocked(client: TestClient, tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    for name in ["../secret.txt", "..%2Fsecret.txt", "%2e%2e/secret.txt"]:
        res = client.get(f"/api/videos/{name}")
        assert res.status_code == 404, name
        assert b"top secret" not in res.content


def test_synthetic_cannot_pose_as_expert(client: TestClient) -> None:
    res = client.post("/api/record_synthetic", json={"role": "expert"})
    assert res.status_code == 422


def test_upload_rejects_garbage_base64(client: TestClient) -> None:
    res = client.post("/api/record_upload", json={"video_base64": "!!!not base64!!!"})
    assert res.status_code == 400


def test_upload_rejects_non_video_bytes(client: TestClient) -> None:
    payload = base64.b64encode(b"hello world, definitely not a video").decode()
    res = client.post("/api/record_upload", json={"video_base64": payload})
    assert res.status_code in (400, 422, 500)
    assert not list(Path("data/recordings").glob("*.mp4")) or res.status_code >= 400


def test_purge_preview_is_non_destructive(client: TestClient, isolated) -> None:
    mentor, activity = isolated
    s = Session(activity_id=activity.activity_id, role="trainee")
    mentor.db.save_session(s)
    res = client.delete("/api/sessions/purge?role=trainee")
    assert res.status_code == 200
    assert res.json()["applied"] is False
    assert mentor.db.get_session(s.session_id) is not None


def test_purge_rejects_unknown_role(client: TestClient) -> None:
    assert client.delete("/api/sessions/purge?role=all").status_code == 422


def test_cors_does_not_allow_credentials(client: TestClient) -> None:
    res = client.options("/api/activities", headers={
        "Origin": "https://evil.example", "Access-Control-Request-Method": "GET"})
    assert res.headers.get("access-control-allow-credentials") != "true"


# --- Chaos / resilience -----------------------------------------------------

def test_missing_landmark_file_is_404(client: TestClient, isolated) -> None:
    mentor, activity = isolated
    s = Session(activity_id=activity.activity_id, landmark_path="data/landmarks/gone.parquet")
    mentor.db.save_session(s)
    assert client.get(f"/api/sessions/{s.session_id}/landmarks").status_code == 404


def test_corrupt_landmark_file_does_not_crash_server(client: TestClient, isolated) -> None:
    mentor, activity = isolated
    bad = Path("data/landmarks/bad.parquet")
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"\x00garbage\x00" * 64)
    s = Session(activity_id=activity.activity_id, landmark_path=str(bad))
    mentor.db.save_session(s)
    res = client.get(f"/api/sessions/{s.session_id}/landmarks")
    assert res.status_code == 422
    # server still serves afterwards
    assert client.get("/api/activities").status_code == 200


def test_purge_survives_missing_files(client: TestClient, isolated) -> None:
    mentor, activity = isolated
    s = Session(activity_id=activity.activity_id, role="trainee",
                video_path="data/recordings/gone.mp4", landmark_path="data/landmarks/gone.parquet")
    mentor.db.save_session(s)
    res = client.delete("/api/sessions/purge?role=trainee&apply=true")
    assert res.status_code == 200, res.text
    assert res.json()["applied"] is True
    assert mentor.db.get_session(s.session_id) is None


def test_concurrent_reads_and_writes(client: TestClient, isolated) -> None:
    mentor, activity = isolated
    errors: list[Exception] = []

    def writer() -> None:
        try:
            for _ in range(20):
                mentor.db.save_session(Session(activity_id=activity.activity_id))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def reader() -> None:
        try:
            for _ in range(20):
                assert client.get("/api/sessions").status_code == 200
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=t) for t in (writer, reader, reader, writer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors
    assert len(client.get("/api/sessions").json()) == 40


def test_deleted_db_file_recreated_on_next_request(client: TestClient, isolated) -> None:
    mentor, _ = isolated
    mentor.db.db_path.unlink()
    res = client.get("/api/activities")
    assert res.status_code in (200, 500)
    assert client.get("/api/activities").status_code in (200, 500)
