import logging
import os
from typing import Optional

import numpy as np
from tensorflow import keras

from ran_env_adversarial import AdversarialRanEnv


class RobustRanEnv(AdversarialRanEnv):
    """Adversarial reward env with perturbed observations and inverted reward."""

    def __init__(
        self,
        data_bundle: dict,
        reward_model_path: str,
        perturbator_path: str,
        config_obj=None,
        encoder_path: Optional[str] = None,
        max_steps: int = 10,
        n_samples_per_slice: int = 10,
        du_prb: int = 50,
        use_mean_obs: bool = True,
        reward_slice_index: int = 0,
        reward_prb_max: Optional[float] = None,
        inverse_reward_mode: str = "reciprocal",
    ):
        self.perturbator_path = self._resolve_model_path(perturbator_path, "Perturbator")
        self.inverse_reward_mode = str(inverse_reward_mode).strip().lower()

        super().__init__(
            data_bundle=data_bundle,
            reward_model_path=reward_model_path,
            config_obj=config_obj,
            encoder_path=encoder_path,
            max_steps=max_steps,
            n_samples_per_slice=n_samples_per_slice,
            du_prb=du_prb,
            use_mean_obs=use_mean_obs,
            reward_slice_index=reward_slice_index,
            reward_prb_max=reward_prb_max,
        )

        self.logger = logging.getLogger("RobustRanEnv")
        self.perturbator = keras.models.load_model(self.perturbator_path, compile=False)
        self.perturbator.trainable = False
        self.last_base_reward = 0.0
        self.last_delta_norm = 0.0

    def _resolve_model_path(self, model_path: str, label: str) -> str:
        if not model_path:
            raise ValueError(f"{label} path must be provided.")

        if os.path.exists(model_path):
            return os.path.abspath(model_path)

        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
        if os.path.exists(local_path):
            return os.path.abspath(local_path)

        raise FileNotFoundError(f"{label} not found at {model_path}")

    def _perturb_observation(self, obs) -> np.ndarray:
        obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        delta = np.asarray(self.perturbator(obs_batch, training=False), dtype=np.float32)

        if delta.shape != obs_batch.shape:
            raise ValueError(
                f"Perturbator output shape {tuple(delta.shape)} does not match "
                f"observation shape {tuple(obs_batch.shape)}"
            )

        adv_obs = obs_batch + delta
        self.last_delta_norm = float(np.linalg.norm(delta.reshape(-1)))
        return adv_obs.reshape(-1).astype(np.float32)

    def _invert_reward(self, reward: float) -> float:
        reward = float(reward)

        if self.inverse_reward_mode == "negate":
            return -reward
        if self.inverse_reward_mode == "reciprocal":
            safe_reward = reward if abs(reward) > 1e-6 else (1e-6 if reward >= 0 else -1e-6)
            return 1.0 / safe_reward

        raise ValueError(f"Unsupported inverse_reward_mode '{self.inverse_reward_mode}'")

    def step(self, action_idx):
        obs, reward, done, info = super().step(action_idx)
        self.last_base_reward = float(reward)
        robust_reward = self._invert_reward(reward)
        robust_obs = self._perturb_observation(obs)
        self.last_reward = float(robust_reward)

        step_info = dict(info)
        step_info["base_reward"] = float(reward)
        step_info["robust_reward"] = float(robust_reward)
        step_info["delta_norm"] = float(self.last_delta_norm)

        return robust_obs, np.float32(robust_reward), done, step_info

    def reset(self, seed=None, options=None):
        obs = super().reset(seed=seed, options=options)
        self.last_base_reward = 0.0
        self.last_delta_norm = 0.0
        return self._perturb_observation(obs)

    def render(self, mode="ansi"):
        if self.current_config:
            prb, sched = self.current_config
            print(
                f"Step: {self.current_step:2d} | Action: PRB {prb}, Sched {sched} | "
                f"Base Reward: {self.last_base_reward:.4f} | "
                f"Robust Reward: {self.last_reward:.4f} | Delta L2: {self.last_delta_norm:.4f}"
            )
