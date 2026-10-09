"""Обучение PPO в CartPole-v1 и экспорт политики в формат NumPy."""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import ENV_ID, EVAL_EPISODES, MODEL_ARCHIVE, NUMPY_POLICY, ROOT_DIR as PROJECT_ROOT, SEED


def extract_numpy_weights(model: Any, output_path: str | Path) -> Path:
    """Извлечь веса двух слоёв политики PPO и записать их в NPZ."""
    import torch

    if model.policy.activation_fn is not torch.nn.Tanh:
        raise RuntimeError("Экспорт остановлен: ожидается активация Tanh в политике PPO")

    policy_modules = list(model.policy.mlp_extractor.policy_net)
    linear_layers = [module for module in policy_modules if isinstance(module, torch.nn.Linear)]
    if len(linear_layers) != 2:
        raise RuntimeError(f"Экспорт остановлен: ожидались два скрытых Linear-слоя, найдено {len(linear_layers)}")
    if len(policy_modules) < 4 or not isinstance(policy_modules[1], torch.nn.Tanh) or not isinstance(policy_modules[3], torch.nn.Tanh):
        raise RuntimeError("Экспорт остановлен: структура скрытых слоёв не соответствует Linear → Tanh → Linear → Tanh")
    output_layer = model.policy.action_net
    if not isinstance(output_layer, torch.nn.Linear) or output_layer.out_features != 2:
        raise RuntimeError("Экспорт остановлен: выходной слой должен выдавать два логита действий")

    layer_1, layer_2 = linear_layers
    arrays = {
        "W1": layer_1.weight.detach().cpu().numpy().astype(np.float32),
        "b1": layer_1.bias.detach().cpu().numpy().astype(np.float32),
        "W2": layer_2.weight.detach().cpu().numpy().astype(np.float32),
        "b2": layer_2.bias.detach().cpu().numpy().astype(np.float32),
        "W3": output_layer.weight.detach().cpu().numpy().astype(np.float32),
        "b3": output_layer.bias.detach().cpu().numpy().astype(np.float32),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return output


def verify_numpy_export(model: Any, policy_path: str | Path, sample_count: int = 1000) -> None:
    """Проверить совпадение жадного действия SB3 и NumPy на 1000 состояниях."""
    from app.policy import NumpyPolicy

    policy = NumpyPolicy.load(policy_path)
    generator = np.random.default_rng(SEED)
    low = np.asarray([-2.4, -4.0, -0.418, -4.0], dtype=np.float32)
    high = np.asarray([2.4, 4.0, 0.418, 4.0], dtype=np.float32)
    states = generator.uniform(low=low, high=high, size=(sample_count, 4)).astype(np.float32)
    mismatches: list[tuple[int, int, int]] = []
    for index, state in enumerate(states):
        sb3_action = int(np.asarray(model.predict(state, deterministic=True)[0]).reshape(-1)[0])
        numpy_action = policy.predict(state)
        if sb3_action != numpy_action:
            mismatches.append((index, sb3_action, numpy_action))
            if len(mismatches) >= 5:
                break
    if mismatches:
        details = ", ".join(f"№{i}: SB3={sb3}, NumPy={numpy}" for i, sb3, numpy in mismatches)
        raise RuntimeError(f"Экспорт политики не прошёл проверку: не совпали действия на состояниях ({details})")
    print(f"Проверка экспорта пройдена: действия совпали на {sample_count} случайных состояниях.")


def train_and_export(
    timesteps: int = 100_000,
    out_dir: str | Path | None = None,
    on_progress: Callable[[int, int, float | None, float | None], None] | None = None,
) -> tuple[Any, Path, Path]:
    """Обучить модель, сохранить архив и экспортировать политику без смены версии."""
    import gymnasium as gym
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    if timesteps < 1:
        raise ValueError("Число шагов обучения должно быть положительным")
    output_dir = Path(out_dir) if out_dir is not None else PROJECT_ROOT / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "ppo_cartpole_v1.0.zip"
    numpy_path = output_dir / "policy_v1.0.npz"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    try:
        torch.use_deterministic_algorithms(True)
    except (RuntimeError, AttributeError) as exc:
        print(f"Предупреждение: не удалось включить строгий детерминизм PyTorch: {exc}")

    environment = gym.make(ENV_ID)
    environment.reset(seed=SEED)
    environment.action_space.seed(SEED)
    start_time = time.monotonic()

    class ProgressCallback(BaseCallback):
        """Публиковать шаги обучения и оценку оставшегося времени."""

        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.last_report = 0

        def _on_step(self) -> bool:
            current = min(int(self.num_timesteps), timesteps)
            if current - self.last_report >= 1000 or current >= timesteps:
                self.last_report = current
                elapsed = max(time.monotonic() - start_time, 0.001)
                eta = elapsed * max(0, timesteps - current) / max(current, 1)
                episodic_rewards = [float(item["r"]) for item in self.model.ep_info_buffer if "r" in item]
                mean_reward = float(np.mean(episodic_rewards)) if episodic_rewards else None
                percent = current * 100.0 / timesteps
                mean_text = f"{mean_reward:.1f}" if mean_reward is not None else "пока нет завершённых эпизодов"
                print(f"Обучение PPO: {current}/{timesteps} шагов ({percent:.1f}%), средняя награда: {mean_text}, осталось примерно {int(eta)} с.")
                if on_progress is not None:
                    on_progress(current, timesteps, mean_reward, eta)
            return True

    try:
        model = PPO(
            "MlpPolicy",
            environment,
            seed=SEED,
            device="cpu",
            verbose=0,
            n_steps=1024,
            batch_size=64,
            policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}, "activation_fn": torch.nn.Tanh},
        )
        model.learn(total_timesteps=timesteps, callback=ProgressCallback(), progress_bar=False)
        model.save(str(archive_path.with_suffix("")))
        exported_path = extract_numpy_weights(model, numpy_path)
        verify_numpy_export(model, exported_path, sample_count=1000)
        return model, archive_path, exported_path
    finally:
        environment.close()


def main() -> int:
    """Разобрать аргументы командной строки и выполнить обучение."""
    parser = argparse.ArgumentParser(description="Обучение PPO-агента в симуляторе CartPole-v1")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Количество шагов обучения, по умолчанию 100000")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "models"), help="Каталог для файлов модели")
    args = parser.parse_args()
    if args.timesteps < 1:
        parser.error("--timesteps должен быть положительным числом")

    output_dir = Path(args.out).expanduser().resolve()
    print(f"Среда: {ENV_ID}; seed: {SEED}; шагов обучения: {args.timesteps}; устройство: CPU")
    try:
        model, archive_path, numpy_path = train_and_export(args.timesteps, output_dir)
        from app.evaluation_service import run_evaluation
        result = run_evaluation(lambda observation: int(model.predict(observation, deterministic=True)[0]))
        mean_reward = result["ppo"]["mean"]
        print(f"Средняя greedy-награда после обучения: {mean_reward:.2f} из 500.")
        if mean_reward < 475:
            print("Предупреждение: качество ниже 475. Попробуйте увеличить --timesteps, например до 200000.")
        print(f"Архив модели: {archive_path}")
        print(f"NumPy-политика: {numpy_path}")
        return 0
    except Exception as exc:
        print(f"Ошибка обучения: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
