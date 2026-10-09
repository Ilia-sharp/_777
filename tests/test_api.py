"""Проверки публичного HTTP-контракта без обучения модели."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from config import MODEL_VERSION


def test_health_contract() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"] == MODEL_VERSION
    assert body["model_loaded"] is True


def test_act_contract_has_exact_keys(tmp_path, monkeypatch) -> None:
    from app import action_logger

    monkeypatch.setattr(action_logger, "LOG_PATH", tmp_path / "api-actions.jsonl")
    with TestClient(app) as client:
        response = client.post("/act", json={"state": [0.01, 0.2, -0.03, -0.4]})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"action", "model_version"}
    assert body["action"] in {0, 1}
    assert body["model_version"] == "ppo-cartpole-v1.0"


def test_invalid_state_length_is_rejected() -> None:
    with TestClient(app) as client:
        response = client.post("/act", json={"state": [0.1, 0.2, 0.3]})
    assert response.status_code == 422
    assert "четыре конечных числа" in response.json()["detail"]


def test_missing_state_is_rejected() -> None:
    with TestClient(app) as client:
        response = client.post("/act", json={})
    assert response.status_code == 422
    assert "state" in response.json()["detail"]


def test_non_finite_state_is_rejected() -> None:
    with TestClient(app) as client:
        nan_response = client.post("/act", content='{"state":[NaN,0,0,0]}', headers={"content-type": "application/json"})
        inf_response = client.post("/act", content='{"state":[Infinity,0,0,0]}', headers={"content-type": "application/json"})
    assert nan_response.status_code == 422
    assert inf_response.status_code == 422


def test_non_finite_reward_is_rejected() -> None:
    with TestClient(app) as client:
        response = client.post("/act", content='{"state":[0,0,0,0],"reward":NaN}', headers={"content-type": "application/json"})
    assert response.status_code == 422


def test_train_is_disabled_by_default(monkeypatch) -> None:
    import app.main as main_module

    monkeypatch.setattr(main_module, "TRAIN_ENABLED", False)
    with TestClient(app) as client:
        response = client.post("/api/train", json={"timesteps": 5000})
    assert response.status_code == 403
    assert "отключено" in response.json()["detail"]


def test_dashboard_and_static_assets_are_served() -> None:
    with TestClient(app) as client:
        page = client.get("/")
        script = client.get("/static/app.js")
        stylesheet = client.get("/static/style.css")
    assert page.status_code == 200
    assert "Nazdratenko" in page.text
    assert script.status_code == 200
    assert "fetch" in script.text
    assert stylesheet.status_code == 200
    assert "prefers-reduced-motion" in stylesheet.text
