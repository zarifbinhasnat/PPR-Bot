"""Smoke tests for the FastAPI app using the TestClient.

`TestClient` runs the app in-process (including lifespan startup) so we can
exercise endpoints without a live server. These tests don't require the index
artifacts: the app boots in NOT-READY mode when they're absent.
"""

from fastapi.testclient import TestClient

from ppr_bot.api.main import app


def test_health_ok():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_readiness_reports_a_boolean():
    with TestClient(app) as client:
        resp = client.get("/readiness")
        assert resp.status_code == 200
        assert "ready" in resp.json()
        assert isinstance(resp.json()["ready"], bool)


def test_history_empty_for_new_session():
    with TestClient(app) as client:
        resp = client.get("/chat/history/never-seen")
        assert resp.status_code == 200
        assert resp.json()["turns"] == []
