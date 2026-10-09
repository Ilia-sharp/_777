"""Оценить уже сохранённую PPO-модель без повторного обучения."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import MODEL_ARCHIVE, SEED


def main() -> int:
    """Запустить оценку фиксированной модели и сохранить результаты."""
    if not MODEL_ARCHIVE.is_file():
        print("Сначала выполните python scripts/train.py", file=sys.stderr)
        return 1

    try:
        from stable_baselines3 import PPO
        from app.evaluation_service import run_evaluation, save_evaluation
        from config import EVALUATION_FILE

        model = PPO.load(str(MODEL_ARCHIVE), device="cpu")
        print(f"Оценка модели: {MODEL_ARCHIVE.name}; seed: {SEED}; режим: greedy")

        def progress(done: int, total: int) -> None:
            print(f"Оценка: {done}/{total} эпизодов ({done * 100 / total:.0f}%).")

        result = run_evaluation(
            lambda observation: int(model.predict(observation, deterministic=True)[0]),
            progress_callback=progress,
        )
        save_evaluation(result, EVALUATION_FILE)

        print("\nРезультаты оценки")
        print(f"{'Агент':<16} {'Среднее':>10} {'Std':>10} {'Min':>8} {'Max':>8} {'Эпизодов':>10}")
        for name in ("ppo", "random"):
            row = result[name]
            label = "PPO (greedy)" if name == "ppo" else "Random"
            print(f"{label:<16} {row['mean']:>10.2f} {row['std']:>10.2f} {row['min']:>8} {row['max']:>8} {len(row['rewards']):>10}")
        print(f"\nPPO лучше случайного агента в {result['improvement_ratio']:.2f} раза.")
        print(f"Результат сохранён: {EVALUATION_FILE}")
        return 0
    except Exception as exc:
        print(f"Ошибка оценки: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
