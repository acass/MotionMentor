"""Integration and unit tests for FastAPI backend server and REST endpoints."""

import pytest
from fastapi.testclient import TestClient

from motion_mentor.server import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_list_activities_endpoint(client: TestClient) -> None:
    res = client.get("/api/activities")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert any(a["name"] == "reach_and_pinch" for a in data)


def test_list_sessions_endpoint(client: TestClient) -> None:
    res = client.get("/api/sessions")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) >= 1

    # Filter by role
    res_expert = client.get("/api/sessions?role=expert")
    assert res_expert.status_code == 200
    assert all(s["role"] == "expert" for s in res_expert.json())


def test_get_session_metadata_and_landmarks(client: TestClient) -> None:
    # First get a valid session ID
    res = client.get("/api/sessions")
    sessions = res.json()
    expert_sess = next((s for s in sessions if s.get("landmark_path")), None)
    if not expert_sess:
        pytest.skip("No session with landmarks recorded in DB")

    sess_id = expert_sess["session_id"]
    res_detail = client.get(f"/api/sessions/{sess_id}")
    assert res_detail.status_code == 200
    assert res_detail.json()["session_id"] == sess_id

    res_lm = client.get(f"/api/sessions/{sess_id}/landmarks")
    assert res_lm.status_code == 200
    frames = res_lm.json()
    assert isinstance(frames, list)
    assert len(frames) > 0
    assert "hands" in frames[0]
    assert "timestamp_ms" in frames[0]


def test_list_references_endpoint(client: TestClient) -> None:
    res = client.get("/api/references")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    if data:
        ref_id = data[0]["reference_id"]
        res_detail = client.get(f"/api/references/{ref_id}")
        assert res_detail.status_code == 200
        ref_payload = res_detail.json()
        assert "profile" in ref_payload
        assert "envelope" in ref_payload


def test_list_assessments_endpoint(client: TestClient) -> None:
    res = client.get("/api/assessments")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)


def test_video_streaming_endpoint(client: TestClient) -> None:
    res = client.get("/api/sessions")
    sessions = res.json()
    video_sess = next((s for s in sessions if s.get("video_path")), None)
    if not video_sess:
        pytest.skip("No video recorded in DB")

    filename = video_sess["video_path"].split("/")[-1]
    # Request first 1024 bytes (Range header)
    headers = {"Range": "bytes=0-1023"}
    res_video = client.get(f"/api/videos/{filename}", headers=headers)
    assert res_video.status_code in (200, 206)
    assert len(res_video.content) > 0


def test_static_dashboard_index(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "MotionMentor" in res.text
    assert "deck-grid" in res.text
    assert 'id="recordRole"' in res.text
    assert 'value="expert"' in res.text
