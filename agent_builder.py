import importlib
import os

import tensorflow as tf
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network


def _load_default_config():
    module_name = os.environ.get("ADVORAN_CONFIG_MODULE", "config")
    return importlib.import_module(module_name)


def _resolve_config(config_obj=None):
    return config_obj if config_obj is not None else _load_default_config()


def create_ppo_agent(train_env, config_obj=None):
    """Construct PPO agent with actor/value networks from config."""
    cfg = _resolve_config(config_obj)
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.ppo_learning_rate)

    observation_spec = train_env.observation_spec()
    action_spec = train_env.action_spec()

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=cfg.ppo_actor_fc_layers,
        activation_fn=tf.keras.activations.tanh,
    )

    value_net = value_network.ValueNetwork(
        observation_spec,
        fc_layer_params=cfg.ppo_value_fc_layers,
        activation_fn=tf.keras.activations.tanh,
    )

    agent = ppo_clip_agent.PPOClipAgent(
        train_env.time_step_spec(),
        action_spec,
        optimizer,
        actor_net=actor_net,
        value_net=value_net,
        entropy_regularization=cfg.ppo_entropy_regularization,
        importance_ratio_clipping=cfg.ppo_importance_ratio_clipping,
        normalize_observations=cfg.ppo_normalize_observations,
        normalize_rewards=cfg.ppo_normalize_rewards,
        use_gae=cfg.ppo_use_gae,
        num_epochs=cfg.ppo_num_epochs,
        debug_summaries=bool(getattr(cfg, "debug_summaries", False)),
        summarize_grads_and_vars=bool(getattr(cfg, "summarize_grads_and_vars", False)),
        greedy_eval=bool(getattr(cfg, "greedy_eval", False)),
    )

    agent.initialize()
    return agent


def create_agent(train_env, algo: str = "ppo", config_obj=None):
    algo = str(algo).lower().strip()
    if algo == "ppo":
        return create_ppo_agent(train_env, config_obj=config_obj)
    raise ValueError(f"Unsupported algo '{algo}'. Currently supported: ppo")
