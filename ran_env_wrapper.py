import importlib
import multiprocessing
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tf_agents.environments import gym_wrapper
from tf_agents.environments import batched_py_environment
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment

from ran_env import RanEnv


global_bundle = None
_bundle_cache: Dict[str, dict] = {}


def _load_default_config():
    module_name = os.environ.get("ADVORAN_CONFIG_MODULE", "config")
    return importlib.import_module(module_name)


def _resolve_config(config_obj=None):
    return config_obj if config_obj is not None else _load_default_config()


def _bundle_key(cfg) -> str:
    dataset = str(getattr(cfg, "dataset_path", ""))
    metrics = ",".join(getattr(cfg, "metric_list_autoencoder", []))
    return f"{dataset}|{metrics}"


def prepare_data_bundle(config_obj=None):
    """Load CSV once and cache grouped features for fast env lookup."""
    global global_bundle
    cfg = _resolve_config(config_obj)
    key = _bundle_key(cfg)

    if key in _bundle_cache:
        global_bundle = _bundle_cache[key]
        return global_bundle

    csv_path = cfg.dataset_path
    print(f"Wrapper: Loading and Bundling CSV from {csv_path}...")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    feature_cache = {0: {}, 1: {}, 2: {}}
    metric_cols = cfg.metric_list_autoencoder
    context_cols = ["slice_prb_norm", "scheduling_policy_norm"]

    if "reward" not in df.columns:
        df["reward"] = 0.0

    missing_context = [c for c in context_cols if c not in df.columns]
    if missing_context:
        context_cols = ["slice_prb", "scheduling_policy"]

    final_feature_order = metric_cols + context_cols + ["reward"]

    for s_id in range(3):
        slice_df = df[df["slice_id"] == s_id]
        grouped = slice_df.groupby(["slice_prb", "scheduling_policy"])
        for (prb, sched), group in grouped:
            feats = group[final_feature_order].values.astype(np.float32)
            feature_cache[s_id][(prb, sched)] = feats

    bundle = {
        "data": feature_cache,
        "num_metrics": len(metric_cols),
    }

    _bundle_cache[key] = bundle
    global_bundle = bundle
    print("Wrapper: Data Bundle prepared successfully.")
    return bundle


def create_gym_env(config_obj=None, data_bundle=None):
    """Create one RanEnv instance."""
    cfg = _resolve_config(config_obj)
    bundle = data_bundle if data_bundle is not None else prepare_data_bundle(cfg)

    return RanEnv(
        data_bundle=bundle,
        config_obj=cfg,
        encoder_path=cfg.encoder_path if os.path.exists(cfg.encoder_path) else None,
        max_steps=cfg.num_steps_per_episode,
        n_samples_per_slice=10,
        du_prb=cfg.du_prb,
    )


def create_wrapped_env(config_obj=None, data_bundle=None):
    return gym_wrapper.GymWrapper(create_gym_env(config_obj=config_obj, data_bundle=data_bundle))


def get_training_env(config_obj=None, num_parallel_override: Optional[int] = None):
    cfg = _resolve_config(config_obj)
    bundle = prepare_data_bundle(cfg)

    max_workers = max(1, multiprocessing.cpu_count() - 2)
    cfg_parallel = int(getattr(cfg, "num_parallel_environments", 1))
    requested = cfg_parallel if num_parallel_override is None else int(num_parallel_override)
    num_parallel = max(1, min(requested, max_workers))

    print(f"Wrapper: Spawning {num_parallel} parallel environments...")

    env_constructors = [
        (lambda cfg=cfg, bundle=bundle: create_wrapped_env(config_obj=cfg, data_bundle=bundle))
        for _ in range(num_parallel)
    ]

    py_env = parallel_py_environment.ParallelPyEnvironment(env_constructors)
    return tf_py_environment.TFPyEnvironment(py_env)


def get_eval_env(config_obj=None):
    cfg = _resolve_config(config_obj)
    bundle = prepare_data_bundle(cfg)
    print("Wrapper: Creating Single Evaluation Environment...")

    base_env = create_wrapped_env(config_obj=cfg, data_bundle=bundle)
    batched_env = batched_py_environment.BatchedPyEnvironment(
        [base_env], multithreading=False
    )
    return tf_py_environment.TFPyEnvironment(batched_env, isolation=False)
