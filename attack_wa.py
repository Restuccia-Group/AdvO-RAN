#!/usr/bin/env python3

import argparse
import glob
import os

import numpy as np
import tensorflow as tf

import agent_builder
import ran_env_wrapper
import config_em_filtered as cfg


DEFAULT_POLICY_DIR_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "saved_policy", "em-max", "em-agent-lp"),
    os.path.join(os.path.dirname(__file__), "saved_policy", "em-max", "em-agent-filtered"),
    os.path.join(os.path.dirname(__file__), "saved_policy", "em-agent-filtered"),
    os.path.join(os.path.dirname(__file__), "saved_policy", "em-agent"),
]


def resolve_policy_dir(policy_dir=None):
    if policy_dir:
        return policy_dir
    for candidate in DEFAULT_POLICY_DIR_CANDIDATES:
        if os.path.isdir(candidate):
            return candidate
    return DEFAULT_POLICY_DIR_CANDIDATES[0]


def _resolve_snapshot_file(policy_dir, prefix, ext, required=True):
    candidates = [os.path.join(policy_dir, f"{prefix}.{ext}")]

    for path in candidates:
        if os.path.exists(path):
            return path

    wildcard = os.path.join(policy_dir, f"{prefix}_*.{ext}")
    matches = sorted(glob.glob(wildcard), key=os.path.getmtime, reverse=True)
    if matches:
        print(
            f"Warning: canonical {prefix}.{ext} not found in {policy_dir}; "
            f"using latest {os.path.basename(matches[0])}"
        )
        return matches[0]

    if required:
        looked_for = ", ".join(os.path.basename(p) for p in candidates)
        raise FileNotFoundError(
            f"Could not find {prefix} snapshot in {policy_dir}. Tried: {looked_for}"
        )
    return candidates[0]


def load_npz_to_vars(npz_path, variables):
    data = np.load(npz_path)
    file_keys = list(data.files)
    count = min(len(file_keys), len(variables))
    for i in range(count):
        arr = data[file_keys[i]]
        var = variables[i]
        if tuple(var.shape) != tuple(arr.shape):
            print(f"Skip raw-order shape mismatch at index {i}: var={tuple(var.shape)} file={tuple(arr.shape)}")
            continue
        var.assign(arr)
    print(f"Loaded {count} vars from {npz_path} by raw order")


def load_snapshot(policy_dir=None):
    policy_dir = resolve_policy_dir(policy_dir)
    return (
        _resolve_snapshot_file(policy_dir, "actor", "npz", required=True),
        _resolve_snapshot_file(policy_dir, "value", "npz", required=True),
        _resolve_snapshot_file(policy_dir, "optimizer", "npz", required=False),
    )


def target_action_id(default=0):
    return default


def decode_action(action_id):
    from config_em_filtered import actions, feasible_prb_allocation_all, scheduling_combos

    a = actions[action_id]
    prb_idx, sched_idx = a[0], scheduling_combos.index(a[1])
    prb_alloc = feasible_prb_allocation_all[prb_idx]
    return prb_alloc, a[1]


def main():
    parser = argparse.ArgumentParser(description="PGD attack on the PPO actor")
    parser.add_argument("--eps", type=float, default=0.3, help="Perturbation budget (L_inf)")
    parser.add_argument("--alpha", type=float, default=0.1, help="Step size per PGD iteration")
    parser.add_argument("--iters", type=int, default=20, help="Number of PGD iterations")
    parser.add_argument("--horizon", type=int, default=20, help="Env steps to run")
    parser.add_argument("--no_attack", action="store_true", help="Run baseline evaluation instead")
    parser.add_argument("--policy_dir", default=None, help="Override saved policy directory")
    parser.add_argument("--target_id", type=int, default=45)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ["ADVORAN_CONFIG_MODULE"] = "config_em_filtered"

    eval_env = ran_env_wrapper.get_eval_env(config_obj=cfg)
    time_step = eval_env.reset()

    agent = agent_builder.create_agent(eval_env, algo="ppo", config_obj=cfg)
    actor_path, value_path, opt_path = load_snapshot(args.policy_dir)

    actor_net = getattr(agent, "actor_net", getattr(agent, "_actor_net"))
    value_net = getattr(agent, "value_net", getattr(agent, "_value_net"))

    load_npz_to_vars(actor_path, actor_net.variables)
    load_npz_to_vars(value_path, value_net.variables)
    if os.path.exists(opt_path):
        opt_ref = getattr(agent, "_optimizer", None) or getattr(agent, "optimizer", None)
        if opt_ref is not None:
            opt_vars = list(opt_ref.variables())
            load_npz_to_vars(opt_path, opt_vars)

    print(f"Loaded snapshot from {os.path.dirname(actor_path)}")

    target_id = target_action_id(args.target_id)
    def run_step(obs, step_type, is_attack, step_idx, phase):
        adv_obs = tf.identity(obs)
        if is_attack:
            delta = tf.random.uniform(tf.shape(obs), -args.eps * 10, args.eps * 10, dtype=tf.float32)
            initial_state = actor_net.get_initial_state(batch_size=tf.shape(obs)[0])
            for _ in range(args.iters):
                adv_obs_var = tf.Variable(obs + delta)
                with tf.GradientTape() as tape:
                    tape.watch(adv_obs_var)
                    dist, _ = actor_net(adv_obs_var, step_type, initial_state, training=False)
                    log_prob = dist.log_prob(tf.constant([[target_id]], dtype=tf.int32))
                    loss = -log_prob
                grad = tape.gradient(loss, adv_obs_var)
                signed = tf.sign(grad)
                delta += args.alpha * signed
                delta = tf.clip_by_value(delta, -args.eps * 10, args.eps * 10)
                adv_obs = tf.stop_gradient(obs + delta)
        attacked_ts = time_step._replace(observation=adv_obs)
        action_step = agent.policy.action(attacked_ts)
        action_id = int(action_step.action.numpy().flatten()[0])
        prb_alloc, sched_combo = decode_action(action_id)
        linf = float(np.max(np.abs((adv_obs - obs).numpy())))
        prefix = "pgd_attack" if is_attack else phase
        print(
            f"[{prefix} t={step_idx}] action id={action_id}, prb_alloc={prb_alloc}, "
            f"sched={sched_combo}, L_inf={linf:.4f}"
        )
        return action_step

    for t in range(args.horizon):
        obs = tf.convert_to_tensor(time_step.observation, dtype=tf.float32)
        step_type = tf.convert_to_tensor(time_step.step_type, dtype=tf.int32)
        action_step = run_step(obs, step_type, not args.no_attack, t, "attack")
        time_step = eval_env.step(action_step.action)
        if time_step.is_last():
            print(f"[t={t}] episode ended early.")
            time_step = eval_env.reset()

    for t in range(args.horizon):
        obs = tf.convert_to_tensor(time_step.observation, dtype=tf.float32)
        step_type = tf.convert_to_tensor(time_step.step_type, dtype=tf.int32)
        action_step = run_step(obs, step_type, False, t, "post")
        time_step = eval_env.step(action_step.action)
        if time_step.is_last():
            print(f"[post t={t}] episode ended early.")
            time_step = eval_env.reset()

if __name__ == "__main__":
    raise SystemExit(main())
