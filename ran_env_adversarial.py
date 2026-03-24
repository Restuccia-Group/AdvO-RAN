import logging
import os
from typing import Optional

import numpy as np
from tensorflow import keras

from ran_env import RanEnv


class AdversarialRanEnv(RanEnv):
    """RanEnv variant that scores actions with a learned reward model."""

    def __init__(
        self,
        data_bundle: dict,
        reward_model_path: str,
        config_obj=None,
        encoder_path: Optional[str] = None,
        max_steps: int = 10,
        n_samples_per_slice: int = 10,
        du_prb: int = 50,
        use_mean_obs: bool = True,
        reward_slice_index: int = 0,
        reward_prb_max: Optional[float] = None,
    ):
        self.reward_model_path = self._resolve_reward_model_path(reward_model_path)
        self.reward_slice_index = int(reward_slice_index)
        self.reward_prb_max_override = reward_prb_max

        super().__init__(
            data_bundle=data_bundle,
            config_obj=config_obj,
            encoder_path=encoder_path,
            max_steps=max_steps,
            n_samples_per_slice=n_samples_per_slice,
            du_prb=du_prb,
            use_mean_obs=use_mean_obs,
        )

        self.logger = logging.getLogger("AdversarialRanEnv")
        self.reward_model = keras.models.load_model(self.reward_model_path, compile=False)
        self.reward_model.trainable = False
        self.reward_prb_max = self._infer_reward_prb_max()
        self.reward_sched_den = self._infer_reward_sched_den()

        if self.reward_prb_max <= 0:
            raise ValueError("reward_prb_max must be positive.")
        if self.reward_sched_den <= 0:
            raise ValueError("reward_sched_den must be positive.")

        self.logger.info(
            "Loaded adversarial reward model from %s (slice=%d, prb_max=%.4f, sched_den=%.4f)",
            self.reward_model_path,
            self.reward_slice_index,
            self.reward_prb_max,
            self.reward_sched_den,
        )

    def _resolve_reward_model_path(self, reward_model_path: str) -> str:
        if not reward_model_path:
            raise ValueError("reward_model_path must be provided.")

        if os.path.exists(reward_model_path):
            return os.path.abspath(reward_model_path)

        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), reward_model_path)
        if os.path.exists(local_path):
            return os.path.abspath(local_path)

        raise FileNotFoundError(f"Reward model not found at {reward_model_path}")

    def _iter_prb_candidates(self):
        for slice_cache in self.data_cache.values():
            for prb, _ in slice_cache.keys():
                yield float(prb)
        for prb_alloc, _ in self.valid_actions:
            for prb in np.asarray(prb_alloc, dtype=np.float32).reshape(-1):
                yield float(prb)
        if self.du_prb:
            yield float(self.du_prb)

    def _iter_sched_candidates(self):
        for slice_cache in self.data_cache.values():
            for _, sched in slice_cache.keys():
                yield float(sched)
        for _, sched_alloc in self.valid_actions:
            for sched in np.asarray(sched_alloc, dtype=np.float32).reshape(-1):
                yield float(sched)

    def _infer_reward_prb_max(self) -> float:
        if self.reward_prb_max_override is not None:
            return float(self.reward_prb_max_override)

        prb_candidates = list(self._iter_prb_candidates())
        if not prb_candidates:
            return 1.0
        return max(prb_candidates)

    def _infer_reward_sched_den(self) -> float:
        sched_candidates = list(self._iter_sched_candidates())
        if not sched_candidates:
            return 1.0
        return max(1.0, max(sched_candidates))

    def _build_reward_features(self, prb_alloc, sched_alloc) -> np.ndarray:
        prb_arr = np.asarray(prb_alloc, dtype=np.float32).reshape(-1)
        sched_arr = np.asarray(sched_alloc, dtype=np.float32).reshape(-1)
        if prb_arr.size == 0 or sched_arr.size == 0:
            raise ValueError("Action allocations must not be empty.")

        max_idx = min(prb_arr.size, sched_arr.size) - 1
        reward_idx = min(max(self.reward_slice_index, 0), max_idx)

        return np.array(
            [[
                float(prb_arr[reward_idx]) / self.reward_prb_max,
                float(sched_arr[reward_idx]) / self.reward_sched_den,
            ]],
            dtype=np.float32,
        )

    def _predict_reward_model(self, prb_alloc, sched_alloc) -> float:
        reward_features = self._build_reward_features(prb_alloc, sched_alloc)
        reward_tensor = self.reward_model(reward_features, training=False)
        return float(np.asarray(reward_tensor).reshape(-1)[0])

    def _get_observation_and_reward(self, prb_alloc, sched_alloc):
        final_obs, _ = super()._get_observation_and_reward(prb_alloc, sched_alloc)
        reward = self._predict_reward_model(prb_alloc, sched_alloc)
        return final_obs, reward
