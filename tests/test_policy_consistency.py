"""Проверка совпадения NumPy-инференса с сохранённой политикой SB3."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from app.evaluation_service import run_evaluation
from app.policy import NumpyPolicy
from config import MODEL_ARCHIVE, NUMPY_POLICY, SEED

SB3_AVAILABLE = importlib.util.find_spec("stable_baselines3") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not TORCH_AVAILABLE or not MODEL_ARCHIVE.is_file() or not NUMPY_POLICY.is_file(), reason="Для сравнения весов нужны PyTorch и файлы модели")
def test_archive_actor_weights_match_numpy_export() -> None:
    """Убедиться, что актёр в SB3-архиве и NPZ содержит одинаковые веса."""
    import io
    import zipfile
    import torch

    assert MODEL_ARCHIVE.is_file()
    assert NUMPY_POLICY.is_file()
    with zipfile.ZipFile(MODEL_ARCHIVE) as archive:
        payload = torch.load(io.BytesIO(archive.read("policy.pth")), map_location="cpu", weights_only=True)
    state = payload["policy"]
    policy = NumpyPolicy.load(NUMPY_POLICY)
    pairs = {
        "mlp_extractor.policy_net.0.weight": policy.W1,
        "mlp_extractor.policy_net.0.bias": policy.b1,
        "mlp_extractor.policy_net.2.weight": policy.W2,
        "mlp_extractor.policy_net.2.bias": policy.b2,
        "action_net.weight": policy.W3,
        "action_net.bias": policy.b3,
    }
    for name, expected in pairs.items():
        np.testing.assert_array_equal(state[name].cpu().numpy(), expected)


@pytest.mark.skipif(not SB3_AVAILABLE or not MODEL_ARCHIVE.is_file() or not NUMPY_POLICY.is_file(), reason="Для проверки нужен Stable-Baselines3 и файлы модели")
def test_numpy_and_sb3_choose_same_action_for_1000_states() -> None:
    from stable_baselines3 import PPO

    model = PPO.load(str(MODEL_ARCHIVE), device="cpu")
    policy = NumpyPolicy.load(NUMPY_POLICY)
    generator = np.random.default_rng(SEED)
    low = np.asarray([-2.4, -4.0, -0.418, -4.0], dtype=np.float32)
    high = np.asarray([2.4, 4.0, 0.418, 4.0], dtype=np.float32)
    for observation in generator.uniform(low=low, high=high, size=(1000, 4)).astype(np.float32):
        sb3_action = int(np.asarray(model.predict(observation, deterministic=True)[0]).reshape(-1)[0])
        assert policy.predict(observation) == sb3_action


@pytest.mark.skipif(not SB3_AVAILABLE or not MODEL_ARCHIVE.is_file() or not NUMPY_POLICY.is_file(), reason="Для проверки нужен Stable-Baselines3 и файлы модели")
def test_numpy_and_sb3_evaluation_rewards_match() -> None:
    from stable_baselines3 import PPO

    model = PPO.load(str(MODEL_ARCHIVE), device="cpu")
    policy = NumpyPolicy.load(NUMPY_POLICY)
    sb3_result = run_evaluation(lambda observation: int(model.predict(observation, deterministic=True)[0]))
    numpy_result = run_evaluation(policy.predict)
    assert sb3_result["ppo"]["rewards"] == numpy_result["ppo"]["rewards"]
    assert sb3_result["random"]["rewards"] == numpy_result["random"]["rewards"]
    for key in ("mean", "std", "min", "max"):
        assert sb3_result["ppo"][key] == numpy_result["ppo"][key]
        assert sb3_result["random"][key] == numpy_result["random"][key]
