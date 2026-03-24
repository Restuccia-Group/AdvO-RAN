import importlib
import multiprocessing
import os
from typing import Optional

from tf_agents.environments import batched_py_environment
from tf_agents.environments import gym_wrapper
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment

from ran_env_adversarial import AdversarialRanEnv
from ran_env_wrapper import prepare_data_bundle


def _load_default_config():
    module_name = os.environ.get("ADVORAN_CONFIG_MODULE", "config")
    return importlib.import_module(module_name)


def _resolve_config(config_obj=None):
    return config_obj if config_obj is not None else _load_default_config()


def create_gym_env(
    config_obj=None,
    data_bundle=None,
    reward_model_path: str = "reward_model.h5",
    reward_slice_index: int = 0,
    reward_prb_max: Optional[float] = None,
):
    cfg = _resolve_config(config_obj)
    bundle = data_bundle if data_bundle is not None else prepare_data_bundle(cfg)

    return AdversarialRanEnv(
        data_bundle=bundle,
        reward_model_path=reward_model_path,
        config_obj=cfg,
        encoder_path=cfg.encoder_path if os.path.exists(cfg.encoder_path) else None,
        max_steps=cfg.num_steps_per_episode,
        n_samples_per_slice=10,
        du_prb=cfg.du_prb,
        reward_slice_index=reward_slice_index,
        reward_prb_max=reward_prb_max,
    )


def create_wrapped_env(
    config_obj=None,
    data_bundle=None,
    reward_model_path: str = "reward_model.h5",
    reward_slice_index: int = 0,
    reward_prb_max: Optional[float] = None,
):
    return gym_wrapper.GymWrapper(
        create_gym_env(
            config_obj=config_obj,
            data_bundle=data_bundle,
            reward_model_path=reward_model_path,
            reward_slice_index=reward_slice_index,
            reward_prb_max=reward_prb_max,
        )
    )


def get_training_env(
    config_obj=None,
    num_parallel_override: Optional[int] = None,
    reward_model_path: str = "reward_model.h5",
    reward_slice_index: int = 0,
    reward_prb_max: Optional[float] = None,
):
    cfg = _resolve_config(config_obj)
    bundle = prepare_data_bundle(cfg)

    max_workers = max(1, multiprocessing.cpu_count() - 2)
    cfg_parallel = int(getattr(cfg, "num_parallel_environments", 1))
    requested = cfg_parallel if num_parallel_override is None else int(num_parallel_override)
    num_parallel = max(1, min(requested, max_workers))

    print(f"Adversarial Wrapper: Spawning {num_parallel} parallel environments...")

    env_constructors = [
        (
            lambda cfg=cfg, bundle=bundle, reward_model_path=reward_model_path,
                   reward_slice_index=reward_slice_index, reward_prb_max=reward_prb_max:
                create_wrapped_env(
                    config_obj=cfg,
                    data_bundle=bundle,
                    reward_model_path=reward_model_path,
                    reward_slice_index=reward_slice_index,
                    reward_prb_max=reward_prb_max,
                )
        )
        for _ in range(num_parallel)
    ]

    py_env = parallel_py_environment.ParallelPyEnvironment(env_constructors)
    return tf_py_environment.TFPyEnvironment(py_env)


def get_eval_env(
    config_obj=None,
    reward_model_path: str = "reward_model.h5",
    reward_slice_index: int = 0,
    reward_prb_max: Optional[float] = None,
):
    cfg = _resolve_config(config_obj)
    bundle = prepare_data_bundle(cfg)
    print("Adversarial Wrapper: Creating Single Evaluation Environment...")

    base_env = create_wrapped_env(
        config_obj=cfg,
        data_bundle=bundle,
        reward_model_path=reward_model_path,
        reward_slice_index=reward_slice_index,
        reward_prb_max=reward_prb_max,
    )
    batched_env = batched_py_environment.BatchedPyEnvironment(
        [base_env], multithreading=False
    )
    return tf_py_environment.TFPyEnvironment(batched_env, isolation=False)
