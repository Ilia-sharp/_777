"""Лёгкий инференс политики на NumPy без импорта PyTorch."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config import NUMPY_POLICY


@dataclass(frozen=True)
class NumpyPolicy:
    """Два скрытых слоя Tanh и выходной линейный слой политики PPO."""

    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    W3: np.ndarray
    b3: np.ndarray

    @classmethod
    def load(cls, path: str | Path = NUMPY_POLICY) -> "NumpyPolicy":
        """Загрузить сохранённые веса политики из NPZ-файла."""
        with np.load(Path(path), allow_pickle=False) as weights:
            required = {"W1", "b1", "W2", "b2", "W3", "b3"}
            missing = required.difference(weights.files)
            if missing:
                raise ValueError(f"В файле политики отсутствуют веса: {', '.join(sorted(missing))}")
            policy = cls(*(np.asarray(weights[key], dtype=np.float32) for key in ("W1", "b1", "W2", "b2", "W3", "b3")))
        policy._validate_shapes()
        return policy

    def _validate_shapes(self) -> None:
        """Проверить размеры слоёв, используемые средой CartPole."""
        expected = {
            "W1": (64, 4), "b1": (64,),
            "W2": (64, 64), "b2": (64,),
            "W3": (2, 64), "b3": (2,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"Некорректный размер {name}: ожидался {shape}, получен {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"В весах {name} найдены нечисловые значения")

    def logits(self, state: Any) -> np.ndarray:
        """Вернуть два логита для переданного состояния."""
        observation = np.asarray(state, dtype=np.float32)
        if observation.shape != (4,) or not np.isfinite(observation).all():
            raise ValueError("Состояние должно содержать ровно четыре конечных числа")
        hidden_1 = np.tanh(self.W1 @ observation + self.b1)
        hidden_2 = np.tanh(self.W2 @ hidden_1 + self.b2)
        return self.W3 @ hidden_2 + self.b3

    def predict(self, state: Any) -> int:
        """Вернуть жадное действие: 0 — влево, 1 — вправо."""
        return int(np.argmax(self.logits(state)))


# Глобальная ссылка задаётся один раз во время старта приложения.
_loaded_policy: NumpyPolicy | None = None
_policy_error: str | None = None


def load_policy(path: str | Path = NUMPY_POLICY) -> NumpyPolicy | None:
    """Однократно загрузить политику; отсутствие файла не мешает старту API."""
    global _loaded_policy, _policy_error
    if _loaded_policy is not None:
        return _loaded_policy
    try:
        _loaded_policy = NumpyPolicy.load(path)
        _policy_error = None
    except (OSError, ValueError, KeyError) as exc:
        _loaded_policy = None
        _policy_error = str(exc)
    return _loaded_policy


def get_policy() -> NumpyPolicy | None:
    """Получить уже загруженную политику."""
    return _loaded_policy


def get_policy_error() -> str | None:
    """Получить локальное описание ошибки загрузки для журнала сервера."""
    return _policy_error


def set_policy_for_tests(policy: NumpyPolicy | None) -> None:
    """Установить политику в тестах без обращения к файлу модели."""
    global _loaded_policy, _policy_error
    _loaded_policy = policy
    _policy_error = None
