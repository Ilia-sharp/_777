"""Потокобезопасный журнал успешных обращений к агенту."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from config import ACTIONS_LOG

logger = logging.getLogger(__name__)
LOG_PATH = ACTIONS_LOG
_LOG_LOCK = threading.Lock()


def append_action(record: dict[str, Any]) -> bool:
    """Добавить одну запись в JSONL; сбой журнала не ломает API."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        with _LOG_LOCK:
            with LOG_PATH.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
                stream.flush()
        return True
    except (OSError, TypeError, ValueError) as exc:
        logger.exception("Не удалось записать запрос в журнал: %s", exc)
        return False


def read_last(limit: int = 10) -> list[dict[str, Any]]:
    """Прочитать последние корректные строки журнала в хронологическом порядке."""
    safe_limit = max(1, min(int(limit), 100))
    try:
        with _LOG_LOCK:
            if not LOG_PATH.exists():
                return []
            lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
        rows: list[dict[str, Any]] = []
        for line in reversed(lines):
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
            except json.JSONDecodeError:
                continue
            if len(rows) >= safe_limit:
                break
        return list(reversed(rows))
    except OSError as exc:
        logger.exception("Не удалось прочитать журнал запросов: %s", exc)
        return []
