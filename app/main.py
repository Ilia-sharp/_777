"""FastAPI-сервис агента CartPole с веб-панелью и журналом."""
from __future__ import annotations

import logging
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from config import (
    EVAL_EPISODES,
    EVALUATION_FILE,
    MODEL_VERSION,
    STATIC_DIR,
    TRAIN_ENABLED,
    TRAIN_MAX_TIMESTEPS,
    SEED,
    ENV_ID,
)
from app import progress
from app.action_logger import append_action, read_last
from app.evaluation_service import evaluate_numpy_policy
from app.policy import get_policy, get_policy_error, load_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Подготовить каталоги и один раз загрузить лёгкую NumPy-политику."""
    for directory in (STATIC_DIR, EVALUATION_FILE.parent):
        directory.mkdir(parents=True, exist_ok=True)
    load_policy()
    if get_policy() is None:
        logger.error("Политика не загружена: %s", get_policy_error() or "файл не найден")
    else:
        logger.info("NumPy-политика %s загружена", MODEL_VERSION)
    yield


app = FastAPI(
    title="Агент CartPole как сервис",
    description="Учебный сервис агента PPO, обученного только в симуляторе CartPole-v1.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), check_dir=False), name="static")


class ActRequest(BaseModel):
    """Входные данные агента: четыре координаты состояния и необязательная награда."""

    model_config = ConfigDict(extra="ignore")
    state: list[float]
    reward: float | None = None

    @field_validator("state")
    @classmethod
    def validate_state(cls, value: list[float]) -> list[float]:
        if len(value) != 4:
            raise ValueError("state должен содержать ровно 4 числа")
        if any(not isinstance(item, (int, float)) or not __import__("math").isfinite(item) for item in value):
            raise ValueError("все значения state должны быть конечными числами")
        return [float(item) for item in value]

    @field_validator("reward")
    @classmethod
    def validate_reward(cls, value: float | None) -> float | None:
        if value is not None and not __import__("math").isfinite(value):
            raise ValueError("reward должен быть конечным числом или null")
        return value


class ActResponse(BaseModel):
    """Строгий публичный контракт POST /act."""

    action: int
    model_version: str


class TrainRequest(BaseModel):
    """Параметры короткого кандидатного обучения."""

    timesteps: int = Field(default=30000, ge=5000, le=50000)


class SimulationResetRequest(BaseModel):
    """Запрос на сброс индивидуального эпизода CartPole."""

    session_id: str = Field(min_length=1, max_length=100)
    seed: int = Field(default=SEED, ge=0, le=2_147_483_647)


class SimulationCloseRequest(BaseModel):
    """Запрос на освобождение остановленной браузерной сессии."""

    session_id: str = Field(min_length=1, max_length=100)


class SimulationStepRequest(BaseModel):
    """Несколько последовательных шагов текущего эпизода."""

    session_id: str = Field(min_length=1, max_length=100)
    steps: int = Field(default=4, ge=1, le=8)
    mode: Literal["agent", "manual"] = "agent"
    action: int | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_manual_action(self) -> "SimulationStepRequest":
        if self.mode == "manual" and self.action not in {0, 1}:
            raise ValueError("Для ручного режима выберите действие 0 или 1")
        return self


# Состояния принадлежат отдельным вкладкам браузера и не смешиваются.
_SIMULATIONS: dict[str, dict[str, Any]] = {}
_SIM_LOCK = threading.Lock()


class _FallbackCartPole:
    """Совместимая реализация уравнений CartPole для локального режима без Gymnasium."""

    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masscart + masspole
    length = 0.5
    polemass_length = masspole * length
    force_mag = 10.0
    tau = 0.02
    x_threshold = 2.4
    theta_threshold_radians = 12 * 2 * 3.141592653589793 / 360

    def __init__(self) -> None:
        import numpy as np
        self.np = np
        self.state: list[float] | None = None
        self.steps = 0
        self.rng = np.random.default_rng()

    def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        self.rng = self.np.random.default_rng(seed)
        self.state = self.rng.uniform(low=-0.05, high=0.05, size=(4,)).tolist()
        self.steps = 0
        return self.np.asarray(self.state, dtype=self.np.float32), {}

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        assert self.state is not None
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if int(action) == 1 else -self.force_mag
        costheta = self.np.cos(theta)
        sintheta = self.np.sin(theta)
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Порядок обновления соответствует Euler-интегратору CartPole-v1.
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = [float(x), float(x_dot), float(theta), float(theta_dot)]
        self.steps += 1
        terminated = bool(x < -self.x_threshold or x > self.x_threshold or theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians)
        truncated = self.steps >= 500
        return self.np.asarray(self.state, dtype=self.np.float32), 1.0, terminated, truncated, {}

    def close(self) -> None:
        return None


def _create_simulation_env() -> tuple[Any, str]:
    """Предпочитать настоящую Gymnasium-среду; иметь fallback только для офлайн-разработки."""
    try:
        import gymnasium as gym

        return gym.make(ENV_ID), "Gymnasium CartPole-v1"
    except ImportError:
        logger.warning("Gymnasium не установлена; используется совместимая локальная реализация CartPole")
        return _FallbackCartPole(), "CartPole: совместимый режим офлайн"


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, _exc: RequestValidationError) -> JSONResponse:
    """Не раскрывать английские технические сообщения Pydantic."""
    return JSONResponse(
        status_code=422,
        content={"detail": "Некорректный запрос. state должен содержать ровно четыре конечных числа; reward — конечное число или null."},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    """Возвращать клиенту понятное русское сообщение."""
    detail = exc.detail if isinstance(exc.detail, str) else "Запрос не может быть выполнен"
    return JSONResponse(status_code=exc.status_code, content={"detail": detail}, headers=exc.headers)


@app.exception_handler(Exception)
async def unexpected_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Записывать подробности в stderr, но не отдавать traceback наружу."""
    logger.exception("Необработанная ошибка API: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Внутренняя ошибка сервиса. Подробности записаны в журнал сервера."})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _run_evaluation_job() -> None:
    """Выполнить оценку в рабочем потоке, не блокируя ответы API."""
    try:
        policy = get_policy()
        if policy is None:
            progress.fail("Оценка недоступна: файл обученной политики не найден.")
            return

        def on_progress(done: int, total: int) -> None:
            progress.update(
                stage="evaluation",
                done=done,
                total=total,
                message=f"Оценка качества: эпизод {done} из {total}",
            )

        result = evaluate_numpy_policy(policy, progress_callback=on_progress)
        progress.finish(result, message="Оценка качества завершена")
    except Exception as exc:
        logger.exception("Не удалось выполнить оценку: %s", exc)
        progress.fail("Не удалось выполнить оценку. Подробности записаны в журнал сервера.")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Отдать одностраничную панель управления."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    """Проверить доступность сервиса и наличие обученной политики."""
    return {"status": "ok", "model_version": MODEL_VERSION, "model_loaded": get_policy() is not None}


@app.post("/act", response_model=ActResponse, response_model_exclude_none=True)
def act(request: ActRequest) -> dict[str, Any]:
    """Получить жадное действие и записать успешное обращение в JSONL."""
    policy = get_policy()
    if policy is None:
        raise HTTPException(status_code=503, detail="Обученная политика не загружена. Проверьте наличие файла весов модели.")

    action = policy.predict(request.state)
    response = {"action": int(action), "model_version": MODEL_VERSION}
    record = {
        "time": _utc_now(),
        "state": request.state,
        "action": int(action),
        "reward": request.reward,
        "model_version": MODEL_VERSION,
    }
    append_action(record)
    return response


@app.get("/api/logs")
def api_logs(limit: int = Query(default=10, ge=1, le=100)) -> dict[str, Any]:
    """Получить последние строки журнала."""
    items = read_last(limit)
    return {"items": items, "count": len(items)}


@app.get("/api/progress")
def api_progress() -> dict[str, Any]:
    """Вернуть актуальный статус фоновой операции."""
    state = progress.snapshot()
    if state["job_type"] is None:
        state["stage"] = "idle"
    state["train_enabled"] = TRAIN_ENABLED
    return state


@app.post("/api/sim/reset")
def reset_simulation(request: SimulationResetRequest) -> dict[str, Any]:
    """Создать новый настоящий эпизод CartPole и вернуть начальное состояние."""
    env, engine = _create_simulation_env()
    try:
        observation, _info = env.reset(seed=request.seed)
        state = [float(value) for value in observation]
        with _SIM_LOCK:
            previous = _SIMULATIONS.pop(request.session_id, None)
            if previous is not None:
                try:
                    previous["env"].close()
                except Exception:
                    logger.debug("Не удалось закрыть предыдущий эпизод симуляции", exc_info=True)
            # Не позволяем долгоживущей странице создать бесконечное число сред.
            if len(_SIMULATIONS) >= 32:
                oldest_key = next(iter(_SIMULATIONS))
                oldest = _SIMULATIONS.pop(oldest_key)
                try:
                    oldest["env"].close()
                except Exception:
                    pass
            _SIMULATIONS[request.session_id] = {
                "env": env, "state": state, "steps": 0, "total_reward": 0.0,
                "done": False, "seed": request.seed, "engine": engine,
            }
        return {"session_id": request.session_id, "state": state, "steps": 0, "total_reward": 0.0, "done": False, "seed": request.seed, "engine": engine}
    except Exception:
        env.close()
        raise


@app.post("/api/sim/close")
def close_simulation(request: SimulationCloseRequest) -> dict[str, bool]:
    """Освободить среду, когда браузер сбрасывает эпизод или закрывает старую сессию."""
    with _SIM_LOCK:
        simulation = _SIMULATIONS.pop(request.session_id, None)
        if simulation is not None:
            try:
                simulation["env"].close()
            except Exception:
                logger.debug("Не удалось закрыть среду симуляции", exc_info=True)
    return {"closed": simulation is not None}


@app.post("/api/sim/step")
def step_simulation(request: SimulationStepRequest) -> dict[str, Any]:
    """Выполнить реальные шаги среды, получить действия политики и записать журнал."""
    policy = get_policy()
    if policy is None:
        raise HTTPException(status_code=503, detail="Обученная политика не загружена. Сбросьте среду после загрузки модели.")

    frames: list[dict[str, Any]] = []
    with _SIM_LOCK:
        simulation = _SIMULATIONS.get(request.session_id)
        if simulation is None:
            raise HTTPException(status_code=404, detail="Эпизод не найден. Нажмите «Сбросить», чтобы создать новый.")
        if simulation["done"]:
            return {"frames": [], "done": True, "steps": simulation["steps"], "total_reward": simulation["total_reward"], "state": simulation["state"], "engine": simulation["engine"], "message": "Эпизод завершён. Сбросьте среду для нового запуска."}

        for _ in range(request.steps):
            previous_state = list(simulation["state"])
            action = int(request.action) if request.mode == "manual" else int(policy.predict(previous_state))
            observation, reward, terminated, truncated, _info = simulation["env"].step(action)
            state = [float(value) for value in observation]
            simulation["steps"] += 1
            simulation["total_reward"] += float(reward)
            simulation["state"] = state
            simulation["done"] = bool(terminated or truncated)
            append_action({
                "time": _utc_now(), "state": previous_state, "action": action,
                "reward": float(reward), "model_version": MODEL_VERSION,
            })
            frames.append({
                "state": state, "previous_state": previous_state, "action": action,
                "reward": float(reward), "steps": simulation["steps"],
                "total_reward": simulation["total_reward"], "done": simulation["done"],
                "terminated": bool(terminated), "truncated": bool(truncated),
            })
            if simulation["done"]:
                break

        return {
            "frames": frames, "done": simulation["done"], "steps": simulation["steps"],
        "total_reward": simulation["total_reward"], "state": simulation["state"],
        "engine": simulation["engine"], "seed": simulation["seed"],
    }


@app.post("/api/evaluate", status_code=202)
def api_evaluate() -> dict[str, str]:
    """Запустить оценку политики в отдельном потоке."""
    if get_policy() is None:
        raise HTTPException(status_code=503, detail="Оценка недоступна: обученная политика не найдена.")
    try:
        progress.start_job("evaluate", EVAL_EPISODES)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail="Уже выполняется другая операция") from exc
    job_id = uuid.uuid4().hex
    thread = threading.Thread(target=_run_evaluation_job, name=f"evaluation-{job_id[:8]}", daemon=True)
    thread.start()
    return {"job_id": job_id, "message": "Оценка запущена. Следите за прогрессом на странице."}


@app.get("/api/evaluate/result")
def api_evaluation_result() -> dict[str, Any]:
    """Вернуть последний сохранённый результат оценки."""
    if not EVALUATION_FILE.is_file():
        raise HTTPException(status_code=404, detail="Результатов оценки пока нет. Запустите оценку.")
    try:
        import json

        return json.loads(EVALUATION_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.exception("Не удалось прочитать результат оценки: %s", exc)
        raise HTTPException(status_code=500, detail="Не удалось прочитать сохранённый результат оценки.") from exc


def _run_candidate_job(timesteps: int) -> None:
    """Обёртка фонового обучения для безопасного завершения потока."""
    from app.training_service import run_candidate_training

    run_candidate_training(timesteps)


@app.post("/api/train", status_code=202)
def api_train(request: TrainRequest) -> dict[str, str]:
    """Запустить необязательное обучение новой модели-кандидата."""
    if not TRAIN_ENABLED:
        raise HTTPException(
            status_code=403,
            detail="Обучение на этом сервере отключено: модель обучена заранее в симуляторе. Запустите обучение локально или установите TRAIN_ENABLED=true.",
        )
    maximum = min(50000, TRAIN_MAX_TIMESTEPS)
    if request.timesteps > maximum:
        raise HTTPException(status_code=422, detail=f"На этом сервере разрешено не более {maximum} шагов обучения.")
    if get_policy() is None:
        raise HTTPException(status_code=503, detail="Текущая политика не загружена; кандидат не может быть сравнен с основной моделью.")
    try:
        progress.start_job("train", request.timesteps)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail="Уже выполняется другая операция") from exc
    job_id = uuid.uuid4().hex
    thread = threading.Thread(target=_run_candidate_job, args=(request.timesteps,), name=f"training-{job_id[:8]}", daemon=True)
    thread.start()
    return {"job_id": job_id, "message": "Обучение кандидатной модели запущено в отдельном потоке."}
