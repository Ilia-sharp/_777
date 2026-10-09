"""Общее потокобезопасное состояние фоновой операции."""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {
    "job_type": None,
    "stage": "idle",
    "done": 0,
    "total": 0,
    "percent": 0.0,
    "mean_reward": None,
    "eta_seconds": None,
    "message": "Операции не запущены",
    "started_at": None,
    "finished_at": None,
    "error": None,
    "result": None,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def is_running() -> bool:
    """Проверить, выполняется ли операция прямо сейчас."""
    with _LOCK:
        return _STATE["job_type"] is not None and _STATE["finished_at"] is None


def start_job(job_type: str, total: int) -> None:
    """Начать одну задачу или сообщить, что другая уже выполняется."""
    if job_type not in {"train", "evaluate"}:
        raise ValueError("Неизвестный тип операции")
    with _LOCK:
        if _STATE["job_type"] is not None and _STATE["finished_at"] is None:
            raise RuntimeError("Уже выполняется другая операция")
        _STATE.update({
            "job_type": job_type,
            "stage": "initializing",
            "done": 0,
            "total": max(0, int(total)),
            "percent": 0.0,
            "mean_reward": None,
            "eta_seconds": None,
            "message": "Подготовка операции",
            "started_at": _now(),
            "finished_at": None,
            "error": None,
            "result": None,
        })


def update(
    *, stage: str | None = None,
    done: int | None = None,
    total: int | None = None,
    mean_reward: float | None = None,
    eta_seconds: float | None = None,
    message: str | None = None,
) -> None:
    """Обновить доступный интерфейсу прогресс операции."""
    with _LOCK:
        if stage is not None:
            _STATE["stage"] = stage
        if total is not None:
            _STATE["total"] = max(0, int(total))
        if done is not None:
            _STATE["done"] = max(0, int(done))
        denominator = _STATE["total"]
        _STATE["percent"] = round(min(100.0, _STATE["done"] * 100.0 / denominator), 1) if denominator else 0.0
        if mean_reward is not None:
            _STATE["mean_reward"] = round(float(mean_reward), 2)
        if eta_seconds is not None:
            _STATE["eta_seconds"] = max(0, int(eta_seconds))
        if message is not None:
            _STATE["message"] = message


def finish(result: Any = None, message: str = "Операция завершена") -> None:
    """Отметить успешное завершение фоновой задачи."""
    with _LOCK:
        _STATE["stage"] = "done"
        _STATE["done"] = _STATE["total"]
        _STATE["percent"] = 100.0
        _STATE["eta_seconds"] = 0
        _STATE["message"] = message
        _STATE["finished_at"] = _now()
        _STATE["error"] = None
        _STATE["result"] = result


def fail(message: str) -> None:
    """Завершить операцию с безопасным русским сообщением об ошибке."""
    with _LOCK:
        _STATE["stage"] = "error"
        _STATE["message"] = message
        _STATE["finished_at"] = _now()
        _STATE["error"] = message
        _STATE["eta_seconds"] = 0


def snapshot() -> dict[str, Any]:
    """Вернуть независимую копию текущего состояния."""
    with _LOCK:
        return dict(_STATE)
