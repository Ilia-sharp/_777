"""Общая оценка политики PPO и случайного агента в CartPole-v1."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from config import ACTION_MODE, EVAL_EPISODES, ENV_ID, EVALUATION_FILE, MODEL_VERSION, RESULTS_DIR, SEED

ActionFunction = Callable[[np.ndarray], int]
ProgressCallback = Callable[[int, int], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _rollout(env: Any, action_function: ActionFunction, episode_seed: int) -> int:
    observation, _ = env.reset(seed=episode_seed)
    total_reward = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = int(action_function(np.asarray(observation, dtype=np.float32)))
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += int(reward)
    return total_reward


def _summary(rewards: list[int]) -> dict[str, Any]:
    values = np.asarray(rewards, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "min": int(np.min(values)),
        "max": int(np.max(values)),
        "rewards": [int(value) for value in rewards],
    }


def run_evaluation(
    action_function: ActionFunction,
    *,
    episodes: int = EVAL_EPISODES,
    seed: int = SEED,
    progress_callback: ProgressCallback | None = None,
    action_mode: str = ACTION_MODE,
    model_version: str = MODEL_VERSION,
) -> dict[str, Any]:
    """Оценить фиксированную политику и случайного агента на одинаковых seed."""
    import gymnasium as gym

    if episodes < 1:
        raise ValueError("Количество эпизодов должно быть положительным")
    env = gym.make(ENV_ID)
    try:
        ppo_rewards: list[int] = []
        random_rewards: list[int] = []
        env.action_space.seed(seed)
        for index in range(episodes):
            # Для каждой пары используются одинаковые seed среды; шкала прогресса остаётся на 20 эпизодах.
            episode_seed = seed + index
            ppo_rewards.append(_rollout(env, action_function, episode_seed))
            random_rewards.append(_rollout(env, lambda _observation: int(env.action_space.sample()), episode_seed))
            if progress_callback is not None:
                progress_callback(index + 1, episodes)
    finally:
        env.close()

    ppo = _summary(ppo_rewards)
    random = _summary(random_rewards)
    random_mean = random["mean"]
    improvement_ratio = float(ppo["mean"] / random_mean) if random_mean else 0.0
    return {
        "seed": int(seed),
        "episodes": int(episodes),
        "action_mode": action_mode,
        "model_version": model_version,
        "ppo": ppo,
        "random": random,
        "improvement_ratio": improvement_ratio,
        "date": _utc_now(),
    }


def save_evaluation(result: dict[str, Any], path: str | Path = EVALUATION_FILE) -> Path:
    """Сохранить результат оценки в UTF-8 JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output


def evaluate_numpy_policy(
    policy: Any,
    *,
    progress_callback: ProgressCallback | None = None,
    save_path: str | Path = EVALUATION_FILE,
) -> dict[str, Any]:
    """Оценить NumPy-политику и сохранить JSON, используемый API и скриптом."""
    result = run_evaluation(policy.predict, progress_callback=progress_callback)
    save_evaluation(result, save_path)
    return result
