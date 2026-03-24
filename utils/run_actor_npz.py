#!/usr/bin/env python3

import argparse
import json
import os
import random
import sys
from typing import List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

import agent_builder
import ran_env_wrapper


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load an actor.npz snapshot and run it step-by-step in RanEnv."
    )
    parser.add_argument("actor_npz", help="Path to actor.npz")
    parser.add_argument("--config_module", default="config_em_filtered", help="Config module")
    parser.add_argument("--steps", type=int, default=10, help="Number of environment steps to run")
    parser.add_argument(
        "--action_mode",
        choices=["greedy", "sample"],
        default="greedy",
        help="How to pick actions from the actor distribution",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Call RanEnv render() after each step")
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional path to save the collected transitions as JSON",
    )
    return parser.parse_args()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_step_type_shape(step_type: tf.Tensor) -> tf.Tensor:
    return tf.reshape(tf.cast(step_type, tf.int32), [-1])


def ensure_action_shape(actions: tf.Tensor) -> tf.Tensor:
    return tf.reshape(tf.cast(actions, tf.int32), [-1])


def load_npz_to_vars(npz_path: str, variables) -> None:
    data = np.load(npz_path)

    file_keys = list(data.files)
    count = min(len(file_keys), len(variables))
    for idx in range(count):
        arr = data[file_keys[idx]]
        var = variables[idx]
        if tuple(var.shape) != tuple(arr.shape):
            print(
                f"Skip raw-order shape mismatch at index {idx}: "
                f"var={tuple(var.shape)} file={tuple(arr.shape)}"
            )
            continue
        var.assign(arr)
    print(f"Loaded {count} vars from {npz_path} by raw order")


def get_actor_net(agent):
    return getattr(agent, "actor_net", getattr(agent, "_actor_net"))


def build_actor_once(env, actor_net) -> None:
    time_step = env.reset()
    obs = tf.convert_to_tensor(time_step.observation, dtype=tf.float32)
    step_type = ensure_step_type_shape(tf.convert_to_tensor(time_step.step_type, dtype=tf.int32))
    batch_size = tf.shape(obs)[0]
    initial_state = actor_net.get_initial_state(batch_size=batch_size)
    _ = actor_net(obs, step_type, initial_state, training=False)


def get_dist(actor_net, obs: tf.Tensor, step_type: tf.Tensor, training: bool = False):
    obs = tf.cast(obs, tf.float32)
    step_type = ensure_step_type_shape(step_type)
    batch_size = tf.shape(obs)[0]
    initial_state = actor_net.get_initial_state(batch_size=batch_size)
    dist, _ = actor_net(obs, step_type, initial_state, training=training)
    return dist


def action_from_dist(dist, action_mode: str) -> tf.Tensor:
    if action_mode == "sample":
        try:
            action = dist.sample()
        except Exception:
            try:
                logits = dist.logits_parameter()
            except Exception:
                probs = dist.probs_parameter()
                logits = tf.math.log(probs + 1e-8)
            action = tf.random.categorical(logits, 1, dtype=tf.int32)
        return ensure_action_shape(action)

    try:
        action = dist.mode()
    except Exception:
        try:
            action = tf.argmax(dist.logits_parameter(), axis=-1, output_type=tf.int32)
        except Exception:
            action = tf.argmax(dist.probs_parameter(), axis=-1, output_type=tf.int32)
    return ensure_action_shape(action)


def decode_action(cfg, action_id: int):
    action = cfg.actions[action_id]
    return cfg.feasible_prb_allocation_all[action[0]], action[1]


def main():
    args = parse_args()
    set_all_seeds(args.seed)
    os.environ["ADVORAN_CONFIG_MODULE"] = args.config_module

    cfg = __import__(args.config_module)
    env = ran_env_wrapper.get_eval_env(config_obj=cfg)

    agent = agent_builder.create_agent(env, algo="ppo", config_obj=cfg)
    actor_net = get_actor_net(agent)
    build_actor_once(env, actor_net)

    load_npz_to_vars(args.actor_npz, actor_net.variables)

    print(f"Actor loaded from {args.actor_npz}")

    transitions = []
    time_step = env.reset()
    episode_idx = 0

    for step_idx in range(int(args.steps)):
        obs = tf.convert_to_tensor(time_step.observation, dtype=tf.float32)
        step_type = ensure_step_type_shape(tf.convert_to_tensor(time_step.step_type, dtype=tf.int32))
        dist = get_dist(actor_net, obs, step_type, training=False)
        action = action_from_dist(dist, args.action_mode)
        action_id = int(action.numpy().reshape(-1)[0])
        prb_alloc, sched_alloc = decode_action(cfg, action_id)

        next_time_step = env.step(tf.reshape(action, [-1]))
        reward = float(np.asarray(next_time_step.reward.numpy()).reshape(-1)[0])
        done = bool(next_time_step.is_last().numpy()[0])

        row = {
            "episode": episode_idx,
            "step": step_idx,
            "action_id": action_id,
            "prb_alloc": prb_alloc,
            "sched_alloc": sched_alloc,
            "reward": reward,
            "done": done,
            "observation": np.asarray(obs.numpy()[0], dtype=np.float32).tolist(),
            "next_observation": np.asarray(next_time_step.observation.numpy()[0], dtype=np.float32).tolist(),
        }
        transitions.append(row)

        print(
            f"[ep={episode_idx} step={step_idx}] "
            f"action={action_id} prb={prb_alloc} sched={sched_alloc} reward={reward:.4f} done={done}"
        )

        if args.render:
            try:
                env.pyenv.render()
            except Exception:
                pass

        time_step = next_time_step
        if done:
            episode_idx += 1
            time_step = env.reset()

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as file_obj:
            json.dump(transitions, file_obj, indent=2)
        print(f"Saved transitions to {args.output_json}")


if __name__ == "__main__":
    raise SystemExit(main())
