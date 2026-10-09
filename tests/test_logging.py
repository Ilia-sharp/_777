"""Проверка записи JSONL после успешного вызова POST /act."""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app import action_logger
from app.main import app
from config import MODEL_VERSION


def test_act_request_is_written_to_jsonl(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "actions.jsonl"
    monkeypatch.setattr(action_logger, "LOG_PATH", log_path)
    with TestClient(app) as client:
        response = client.post("/act", json={"state": [0.01, 0.2, -0.03, -0.4]})
    assert response.status_code == 200
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert set(record) == {"time", "state", "action", "reward", "model_version"}
    assert record["time"].endswith("Z")
    assert record["state"] == [0.01, 0.2, -0.03, -0.4]
    assert record["action"] in {0, 1}
    assert record["reward"] is None
    assert record["model_version"] == MODEL_VERSION
