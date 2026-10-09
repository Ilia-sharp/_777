"""Единая конфигурация лабораторной работы."""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# Параметры варианта: значения определяются только в этом модуле.
ENV_ID = "CartPole-v1"
SEED = 116
EVAL_EPISODES = 20
ACTION_MODE = "greedy"
MODEL_VERSION = os.getenv("MODEL_VERSION", "ppo-cartpole-v1.0")
REPO_NAME = "SII-LR07-Nazdratenko"
TAG = "v1.0"
PYTHON_MIN = "3.10"

# Пути проекта.
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
LOGS_DIR = ROOT_DIR / "logs"
STATIC_DIR = ROOT_DIR / "static"
MODEL_ARCHIVE = MODELS_DIR / "ppo_cartpole_v1.0.zip"
NUMPY_POLICY = MODELS_DIR / "policy_v1.0.npz"
EVALUATION_FILE = RESULTS_DIR / "evaluation.json"
ACTIONS_LOG = LOGS_DIR / "actions.jsonl"

# Ограничения веб-обучения.
TRAIN_ENABLED = os.getenv("TRAIN_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
TRAIN_MAX_TIMESTEPS = int(os.getenv("TRAIN_MAX_TIMESTEPS", "50000"))
