"""Необязательное фоновое обучение кандидатной модели."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from config import EVAL_EPISODES, MODELS_DIR, NUMPY_POLICY, TRAIN_MAX_TIMESTEPS
from app import progress
from app.evaluation_service import run_evaluation
from app.policy import NumpyPolicy

logger = logging.getLogger(__name__)


def run_candidate_training(timesteps: int) -> None:
    """Обучить отдельного кандидата и сравнить его с текущей политикой."""
    try:
        if timesteps < 5000 or timesteps > min(50000, TRAIN_MAX_TIMESTEPS):
            raise ValueError("Количество шагов обучения выходит за допустимый диапазон")
        if not NUMPY_POLICY.is_file():
            raise RuntimeError("Текущая политика не найдена; сначала обучите основную модель локально")

        from scripts.train import train_and_export

        candidate_dir = MODELS_DIR / "candidate"
        candidate_model, archive_path, numpy_path = train_and_export(
            timesteps,
            candidate_dir,
            on_progress=lambda done, total, mean_reward, eta: progress.update(
                stage="training",
                done=done,
                total=total,
                mean_reward=mean_reward,
                eta_seconds=eta,
                message=f"Обучение кандидатной модели: {done} из {total} шагов",
            ),
        )

        current_policy = NumpyPolicy.load(NUMPY_POLICY)
        candidate_policy = NumpyPolicy.load(numpy_path)
        progress.update(stage="evaluation", done=0, total=EVAL_EPISODES, message="Сравнение кандидата с текущей моделью")

        current_eval = run_evaluation(current_policy.predict)
        candidate_eval = run_evaluation(
            candidate_policy.predict,
            progress_callback=lambda done, total: progress.update(
                stage="evaluation",
                done=done,
                total=total,
                message=f"Проверка кандидатной политики: эпизод {done} из {total}",
            ),
        )
        current_mean = float(current_eval["ppo"]["mean"])
        candidate_mean = float(candidate_eval["ppo"]["mean"])
        outcome: dict[str, Any] = {
            "timesteps": int(timesteps),
            "current_model_mean_reward": current_mean,
            "candidate_mean_reward": candidate_mean,
            "difference": candidate_mean - current_mean,
            "candidate_archive": str(archive_path.relative_to(MODELS_DIR.parent)),
            "candidate_policy": str(numpy_path.relative_to(MODELS_DIR.parent)),
            "note": "Кандидат сохранён отдельно; обслуживаемая модель не заменена.",
        }
        progress.finish(outcome, message="Кандидат обучен и сравнение завершено")
    except Exception as exc:
        logger.exception("Ошибка обучения кандидатной модели: %s", exc)
        progress.fail("Не удалось обучить или проверить кандидатную модель. Подробности записаны в журнал сервера.")
