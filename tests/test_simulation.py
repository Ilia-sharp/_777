"""Проверки живого шага CartPole без запуска обучения."""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app import action_logger
from app.main import app, _SIMULATIONS, _SIM_LOCK


def test_live_simulation_returns_real_frames_and_logs(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "actions.jsonl"
    monkeypatch.setattr(action_logger, "LOG_PATH", log_path)
    with _SIM_LOCK:
        for record in _SIMULATIONS.values():
            record["env"].close()
        _SIMULATIONS.clear()

    with TestClient(app) as client:
        reset = client.post("/api/sim/reset", json={"session_id": "test-live", "seed": 116})
        assert reset.status_code == 200
        reset_body = reset.json()
        assert reset_body["seed"] == 116
        assert reset_body["steps"] == 0
        assert len(reset_body["state"]) == 4
        assert "CartPole" in reset_body["engine"]

        stepped = client.post("/api/sim/step", json={"session_id": "test-live", "steps": 3, "mode": "agent"})
        assert stepped.status_code == 200
        body = stepped.json()
        assert len(body["frames"]) == 3
        assert body["steps"] == 3
        assert body["total_reward"] == 3.0
        assert all(frame["action"] in {0, 1} for frame in body["frames"])
        assert all(len(frame["state"]) == 4 for frame in body["frames"])
        assert all(frame["reward"] == 1.0 for frame in body["frames"])

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 3
    assert all(record["time"].endswith("Z") for record in records)
    assert all(record["reward"] == 1.0 for record in records)

    with _SIM_LOCK:
        item = _SIMULATIONS.pop("test-live", None)
        if item:
            item["env"].close()


def test_manual_simulation_uses_selected_action(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(action_logger, "LOG_PATH", tmp_path / "manual-actions.jsonl")
    with TestClient(app) as client:
        reset = client.post("/api/sim/reset", json={"session_id": "test-manual", "seed": 117})
        assert reset.status_code == 200
        stepped = client.post("/api/sim/step", json={
            "session_id": "test-manual", "steps": 2, "mode": "manual", "action": 1,
        })
        assert stepped.status_code == 200
        assert [frame["action"] for frame in stepped.json()["frames"]] == [1, 1]
    with _SIM_LOCK:
        item = _SIMULATIONS.pop("test-manual", None)
        if item:
            item["env"].close()


def test_step_requires_existing_session() -> None:
    with TestClient(app) as client:
        response = client.post("/api/sim/step", json={"session_id": "missing-session", "steps": 1})
    assert response.status_code == 404
    assert "Нажмите" in response.json()["detail"]


def test_manual_mode_requires_action() -> None:
    with TestClient(app) as client:
        response = client.post("/api/sim/step", json={"session_id": "unused", "mode": "manual", "steps": 1})
    assert response.status_code == 422


def test_close_releases_simulation_session() -> None:
    with TestClient(app) as client:
        reset = client.post("/api/sim/reset", json={"session_id": "test-close", "seed": 118})
        assert reset.status_code == 200
        closed = client.post("/api/sim/close", json={"session_id": "test-close"})
        assert closed.status_code == 200
        assert closed.json() == {"closed": True}
        missing = client.post("/api/sim/step", json={"session_id": "test-close", "steps": 1})
        assert missing.status_code == 404
