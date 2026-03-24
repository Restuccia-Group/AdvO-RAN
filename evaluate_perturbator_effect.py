#!/usr/bin/env python3

import argparse
import glob
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

import agent_builder
import config_em_filtered as cfg
import ran_env_wrapper


DEFAULT_POLICY_DIR_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "saved_policy", "em-max", "em-agent-lp"),
    os.path.join(os.path.dirname(__file__), "saved_policy", "em-max", "em-agent-filtered"),
    os.path.join(os.path.dirname(__file__), "saved_policy", "em-agent-filtered"),
    os.path.join(os.path.dirname(__file__), "saved_policy", "em-agent"),
]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_step_type_shape(step_type: tf.Tensor) -> tf.Tensor:
    return tf.reshape(tf.cast(step_type, tf.int32), [-1])


def ensure_action_shape(actions: tf.Tensor) -> tf.Tensor:
    return tf.reshape(tf.cast(actions, tf.int32), [-1])


def squeeze_policy_tensor(tensor: tf.Tensor) -> tf.Tensor:
    tensor = tf.convert_to_tensor(tensor)
    if tensor.shape.rank == 3 and tensor.shape[1] == 1:
        tensor = tf.squeeze(tensor, axis=1)
    return tensor


def resolve_policy_dir(policy_dir: Optional[str] = None) -> str:
    if policy_dir:
        return policy_dir
    for candidate in DEFAULT_POLICY_DIR_CANDIDATES:
        if os.path.isdir(candidate):
            return candidate
    return DEFAULT_POLICY_DIR_CANDIDATES[0]


def resolve_snapshot_file(
    policy_dir: str,
    prefix: str,
    ext: str,
    required: bool = True,
) -> str:
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
        looked_for = ", ".join(os.path.basename(path) for path in candidates)
        raise FileNotFoundError(
            f"Could not find {prefix} snapshot in {policy_dir}. Tried: {looked_for}"
        )
    return candidates[0]


def load_npz_to_vars(
    npz_path: str,
    variables: List[tf.Variable],
) -> None:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Snapshot not found: {npz_path}")

    data = np.load(npz_path)

    file_keys = list(data.files)
    count = min(len(file_keys), len(variables))
    for index in range(count):
        arr = data[file_keys[index]]
        var = variables[index]
        if tuple(var.shape) != tuple(arr.shape):
            print(
                f"Skip raw-order shape mismatch at index {index}: "
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


def load_actor(env, policy_dir: Optional[str]):
    agent = agent_builder.create_agent(env, algo="ppo", config_obj=cfg)
    actor_net = get_actor_net(agent)
    build_actor_once(env, actor_net)

    policy_dir = resolve_policy_dir(policy_dir)
    actor_path = resolve_snapshot_file(policy_dir, "actor", "npz", required=True)

    load_npz_to_vars(actor_path, actor_net.variables)
    print(f"Loaded actor snapshot from {actor_path}")

    return agent, actor_net, actor_path


def load_perturbator(path: str) -> tf.keras.Model:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Perturbator model not found: {path}")
    model = tf.keras.models.load_model(path, compile=False)
    model.trainable = False
    print(f"Loaded perturbator from {path}")
    return model


def get_dist(actor_net, obs: tf.Tensor, step_type: tf.Tensor, training: bool = False):
    obs = tf.cast(obs, tf.float32)
    step_type = ensure_step_type_shape(step_type)
    batch_size = tf.shape(obs)[0]
    initial_state = actor_net.get_initial_state(batch_size=batch_size)
    dist, _ = actor_net(obs, step_type, initial_state, training=training)
    return dist


def get_probs(dist) -> tf.Tensor:
    try:
        probs = dist.probs_parameter()
    except Exception:
        probs = tf.nn.softmax(dist.logits_parameter(), axis=-1)
    return tf.cast(squeeze_policy_tensor(probs), tf.float32)


def get_logits(dist) -> tf.Tensor:
    try:
        logits = dist.logits_parameter()
    except Exception:
        logits = tf.math.log(get_probs(dist) + 1e-8)
    return tf.cast(squeeze_policy_tensor(logits), tf.float32)


def greedy_action_from_dist(dist) -> tf.Tensor:
    try:
        action = dist.mode()
    except Exception:
        action = tf.argmax(get_logits(dist), axis=-1, output_type=tf.int32)
    return ensure_action_shape(action)


def sample_action_from_dist(dist) -> tf.Tensor:
    try:
        action = dist.sample()
    except Exception:
        logits = get_logits(dist)
        action = tf.random.categorical(logits, 1, dtype=tf.int32)
    return ensure_action_shape(action)


def action_from_dist(dist, action_mode: str) -> tf.Tensor:
    if action_mode == "sample":
        return sample_action_from_dist(dist)
    return greedy_action_from_dist(dist)


def perturb_obs(perturbator: tf.keras.Model, obs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    obs = tf.cast(obs, tf.float32)
    delta = tf.cast(perturbator(obs, training=False), tf.float32)
    if tuple(delta.shape) != tuple(obs.shape):
        raise RuntimeError(
            f"Perturbator output shape {tuple(delta.shape)} does not match "
            f"observation shape {tuple(obs.shape)}"
        )
    return tf.stop_gradient(obs + delta), tf.stop_gradient(delta)


def maybe_seed_env(env, seed: int) -> None:
    try:
        if hasattr(env, "seed"):
            env.seed(seed)
    except Exception:
        pass


def decode_action(action_id: int) -> Tuple[List[int], List[int]]:
    action = cfg.actions[action_id]
    return cfg.feasible_prb_allocation_all[action[0]], action[1]


def format_action(action_id: int) -> str:
    prb_alloc, sched_combo = decode_action(action_id)
    return f"id={action_id} prb={prb_alloc} sched={sched_combo}"


def collect_rollout_batch(
    env,
    actor_net,
    num_steps: int,
    action_mode: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    set_all_seeds(seed)
    maybe_seed_env(env, seed)

    obs_records, step_type_records = [], []
    time_step = env.reset()

    for _ in range(num_steps):
        obs = tf.convert_to_tensor(time_step.observation, dtype=tf.float32)
        step_type = ensure_step_type_shape(tf.convert_to_tensor(time_step.step_type, dtype=tf.int32))

        obs_records.append(obs.numpy()[0])
        step_type_records.append(int(step_type.numpy().reshape(-1)[0]))

        dist = get_dist(actor_net, obs, step_type, training=False)
        action = action_from_dist(dist, action_mode)
        time_step = env.step(tf.reshape(action, [-1]))
        if time_step.is_last():
            time_step = env.reset()

    return (
        np.asarray(obs_records, dtype=np.float32),
        np.asarray(step_type_records, dtype=np.int32),
    )


def _collect_dense_layers_dfs(module) -> List[tf.keras.layers.Dense]:
    collected: List[tf.keras.layers.Dense] = []
    seen_ids = set()

    def visit(obj) -> None:
        obj_id = id(obj)
        if obj_id in seen_ids:
            return
        seen_ids.add(obj_id)

        if isinstance(obj, tf.keras.layers.Dense):
            collected.append(obj)
            return

        for attr in ("_self_tracked_trackables", "layers", "_layers", "_sublayers"):
            sublayers = getattr(obj, attr, None)
            if sublayers is None:
                continue
            try:
                iterator = iter(sublayers)
            except TypeError:
                continue
            for sublayer in iterator:
                visit(sublayer)
            break

    visit(module)
    return collected


def extract_dense_layers(actor_net) -> List[tf.keras.layers.Dense]:
    for sub_attr in ("_encoding_network", "encoding_network"):
        encoder = getattr(actor_net, sub_attr, None)
        if encoder is None:
            continue
        layers = _collect_dense_layers_dfs(encoder)
        for proj_attr in (
            "_output_layers",
            "_projection_networks",
            "_action_projection_layer",
            "_output_tensor_spec",
        ):
            projection = getattr(actor_net, proj_attr, None)
            if projection is not None:
                layers += _collect_dense_layers_dfs(projection)
        if layers:
            return layers
    return _collect_dense_layers_dfs(actor_net)


def dense_forward_trace(
    dense_layers: List[tf.keras.layers.Dense],
    obs: tf.Tensor,
) -> List[tf.Tensor]:
    activations = []
    x = tf.cast(obs, tf.float32)
    for layer in dense_layers:
        x = tf.cast(layer(x), tf.float32)
        activations.append(x)
    return activations


def summarize_layer_effects(
    actor_net,
    obs: tf.Tensor,
    step_type: tf.Tensor,
    dense_layers: List[tf.keras.layers.Dense],
    adv_obs: tf.Tensor,
    verify_tol: float,
) -> Tuple[Optional[List[dict]], Optional[float]]:
    if not dense_layers:
        return None, None

    clean_trace = dense_forward_trace(dense_layers, obs)
    adv_trace = dense_forward_trace(dense_layers, adv_obs)
    clean_logits = get_logits(get_dist(actor_net, obs, step_type, training=False))

    final_logits = clean_trace[-1]
    mae = float(tf.reduce_mean(tf.abs(final_logits - clean_logits)).numpy())
    if mae > verify_tol:
        print(
            f"[layer-trace] skipped: dense-trace final-logit MAE={mae:.4f} "
            f"(tol={verify_tol:.4f})"
        )
        return None, mae

    layer_rows = []
    for idx, (layer, clean_act, adv_act) in enumerate(zip(dense_layers, clean_trace, adv_trace)):
        diff = tf.abs(adv_act - clean_act)
        layer_rows.append(
            {
                "index": idx,
                "name": layer.name,
                "units": int(clean_act.shape[-1]),
                "mean_abs_delta": float(tf.reduce_mean(diff).numpy()),
                "max_abs_delta": float(tf.reduce_max(diff).numpy()),
            }
        )
    return layer_rows, mae


def gather_selected_prob(probs: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
    batch_idx = tf.range(tf.shape(actions)[0], dtype=tf.int32)
    gather_idx = tf.stack([batch_idx, actions], axis=1)
    return tf.gather_nd(probs, gather_idx)


def analyse_action_net_effect(
    actor_net,
    perturbator: tf.keras.Model,
    obs_batch: np.ndarray,
    step_type_batch: np.ndarray,
    collection_action_mode: str,
    dense_layers: List[tf.keras.layers.Dense],
    dense_verify_tol: float,
) -> Dict:
    obs = tf.convert_to_tensor(obs_batch, dtype=tf.float32)
    step_type = ensure_step_type_shape(tf.convert_to_tensor(step_type_batch, dtype=tf.int32))

    adv_obs, delta = perturb_obs(perturbator, obs)

    clean_dist = get_dist(actor_net, obs, step_type, training=False)
    adv_dist = get_dist(actor_net, adv_obs, step_type, training=False)

    clean_probs = get_probs(clean_dist)
    adv_probs = get_probs(adv_dist)
    clean_logits = get_logits(clean_dist)
    adv_logits = get_logits(adv_dist)
    clean_actions = greedy_action_from_dist(clean_dist)
    adv_actions = greedy_action_from_dist(adv_dist)

    delta_linf = tf.reduce_max(tf.abs(delta), axis=-1)
    delta_l2 = tf.norm(delta, ord=2, axis=-1)
    kl = tf.reduce_sum(
        clean_probs * (tf.math.log(clean_probs + 1e-8) - tf.math.log(adv_probs + 1e-8)),
        axis=-1,
    )
    tv = 0.5 * tf.reduce_sum(tf.abs(clean_probs - adv_probs), axis=-1)
    logit_l2 = tf.norm(adv_logits - clean_logits, ord=2, axis=-1)

    clean_top_prob = tf.reduce_max(clean_probs, axis=-1)
    adv_top_prob = tf.reduce_max(adv_probs, axis=-1)
    clean_action_prob_before = gather_selected_prob(clean_probs, clean_actions)
    clean_action_prob_after = gather_selected_prob(adv_probs, clean_actions)

    flip_mask = tf.not_equal(clean_actions, adv_actions)
    top_examples = []
    score = kl + tv + tf.cast(flip_mask, tf.float32)
    top_indices = np.argsort(score.numpy())[::-1][: min(5, obs_batch.shape[0])]
    for idx in top_indices:
        clean_id = int(clean_actions.numpy()[idx])
        adv_id = int(adv_actions.numpy()[idx])
        top_examples.append(
            {
                "index": int(idx),
                "flip": bool(flip_mask.numpy()[idx]),
                "linf": float(delta_linf.numpy()[idx]),
                "l2": float(delta_l2.numpy()[idx]),
                "kl_clean_to_adv": float(kl.numpy()[idx]),
                "tv_distance": float(tv.numpy()[idx]),
                "logit_l2": float(logit_l2.numpy()[idx]),
                "clean_action": clean_id,
                "adv_action": adv_id,
                "clean_action_desc": format_action(clean_id),
                "adv_action_desc": format_action(adv_id),
                "clean_top_prob": float(clean_top_prob.numpy()[idx]),
                "adv_top_prob": float(adv_top_prob.numpy()[idx]),
                "clean_action_prob_before": float(clean_action_prob_before.numpy()[idx]),
                "clean_action_prob_after": float(clean_action_prob_after.numpy()[idx]),
            }
        )

    layer_effects, dense_trace_mae = summarize_layer_effects(
        actor_net=actor_net,
        obs=obs,
        step_type=step_type,
        dense_layers=dense_layers,
        adv_obs=adv_obs,
        verify_tol=dense_verify_tol,
    )

    clean_counts = np.bincount(clean_actions.numpy(), minlength=cfg.n_actions)
    adv_counts = np.bincount(adv_actions.numpy(), minlength=cfg.n_actions)

    return {
        "num_samples": int(obs_batch.shape[0]),
        "analysis_action_mode": "greedy",
        "collection_action_mode": collection_action_mode,
        "obs_dim": int(obs_batch.shape[1]),
        "delta_linf_mean": float(tf.reduce_mean(delta_linf).numpy()),
        "delta_linf_max": float(tf.reduce_max(delta_linf).numpy()),
        "delta_l2_mean": float(tf.reduce_mean(delta_l2).numpy()),
        "action_flip_rate": float(tf.reduce_mean(tf.cast(flip_mask, tf.float32)).numpy()),
        "kl_clean_to_adv_mean": float(tf.reduce_mean(kl).numpy()),
        "tv_distance_mean": float(tf.reduce_mean(tv).numpy()),
        "logit_l2_mean": float(tf.reduce_mean(logit_l2).numpy()),
        "clean_top_prob_mean": float(tf.reduce_mean(clean_top_prob).numpy()),
        "adv_top_prob_mean": float(tf.reduce_mean(adv_top_prob).numpy()),
        "clean_action_prob_drop_mean": float(
            tf.reduce_mean(clean_action_prob_before - clean_action_prob_after).numpy()
        ),
        "dense_trace_mae": dense_trace_mae,
        "layer_effects": layer_effects,
        "top_examples": top_examples,
        "clean_action_histogram": clean_counts.astype(int).tolist(),
        "adv_action_histogram": adv_counts.astype(int).tolist(),
    }


def evaluate_rollout_reward(
    env,
    actor_net,
    perturbator: Optional[tf.keras.Model],
    episodes: int,
    max_steps: int,
    action_mode: str,
    seed: int,
) -> Dict[str, float]:
    set_all_seeds(seed)
    maybe_seed_env(env, seed)

    rewards = []
    lengths = []
    flip_count = 0
    total_steps = 0

    for _ in range(episodes):
        time_step = env.reset()
        ep_reward = 0.0
        ep_len = 0

        while (not time_step.is_last()) and ep_len < max_steps:
            obs = tf.convert_to_tensor(time_step.observation, dtype=tf.float32)
            step_type = ensure_step_type_shape(tf.convert_to_tensor(time_step.step_type, dtype=tf.int32))

            clean_dist = get_dist(actor_net, obs, step_type, training=False)
            clean_action = action_from_dist(clean_dist, action_mode)
            chosen_action = clean_action

            if perturbator is not None:
                adv_obs, _ = perturb_obs(perturbator, obs)
                adv_dist = get_dist(actor_net, adv_obs, step_type, training=False)
                chosen_action = action_from_dist(adv_dist, action_mode)
                flip_count += int(
                    int(chosen_action.numpy().reshape(-1)[0]) != int(clean_action.numpy().reshape(-1)[0])
                )

            time_step = env.step(tf.reshape(chosen_action, [-1]))
            reward = float(np.asarray(time_step.reward.numpy()).reshape(-1)[0])
            ep_reward += reward
            ep_len += 1
            total_steps += 1

            if time_step.is_last():
                break

        rewards.append(ep_reward)
        lengths.append(ep_len)

    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "mean_ep_len": float(np.mean(lengths)) if lengths else 0.0,
        "action_flip_rate": float(flip_count / max(total_steps, 1)),
    }


def print_action_histogram(title: str, histogram: List[int], top_n: int) -> None:
    pairs = [(count, action_id) for action_id, count in enumerate(histogram) if count > 0]
    pairs.sort(reverse=True)
    print(title)
    if not pairs:
        print("  none")
        return
    for count, action_id in pairs[:top_n]:
        print(f"  {count:4d} x {format_action(action_id)}")


def print_report(summary: Dict, top_hist_n: int) -> None:
    action_effect = summary["action_effect"]
    rollout_clean = summary["rollout_clean"]
    rollout_attack = summary["rollout_attack"]

    print("\n" + "=" * 72)
    print("Perturbator Effect On Saved Policy Action Net")
    print("=" * 72)
    print(
        f"policy_dir     : {summary['policy_dir']}\n"
        f"actor_snapshot : {summary['actor_snapshot']}\n"
        f"perturbator    : {summary['perturbator_path']}\n"
        f"samples        : {action_effect['num_samples']}  "
        f"obs_dim={action_effect['obs_dim']}  "
        f"analysis_mode={action_effect['analysis_action_mode']}  "
        f"collection_mode={action_effect['collection_action_mode']}"
    )

    print("\n[Action Net Shift]")
    print(
        f"  delta L_inf mean/max      : {action_effect['delta_linf_mean']:.6f} / "
        f"{action_effect['delta_linf_max']:.6f}"
    )
    print(f"  delta L2 mean             : {action_effect['delta_l2_mean']:.6f}")
    print(f"  action flip rate          : {action_effect['action_flip_rate']:.4f}")
    print(f"  KL(clean || adv) mean     : {action_effect['kl_clean_to_adv_mean']:.6f}")
    print(f"  TV distance mean          : {action_effect['tv_distance_mean']:.6f}")
    print(f"  logit L2 mean             : {action_effect['logit_l2_mean']:.6f}")
    print(
        f"  top prob mean clean/adv   : {action_effect['clean_top_prob_mean']:.6f} / "
        f"{action_effect['adv_top_prob_mean']:.6f}"
    )
    print(f"  clean-action prob drop    : {action_effect['clean_action_prob_drop_mean']:.6f}")

    if action_effect["layer_effects"] is not None:
        print(f"  dense-trace MAE vs logits : {action_effect['dense_trace_mae']:.6f}")
        print("  layer activation drift:")
        for row in action_effect["layer_effects"]:
            print(
                f"    [{row['index']}] {row['name']:<24s} "
                f"mean_abs_delta={row['mean_abs_delta']:.6f} "
                f"max_abs_delta={row['max_abs_delta']:.6f}"
            )
    elif action_effect["dense_trace_mae"] is not None:
        print(f"  dense-trace MAE vs logits : {action_effect['dense_trace_mae']:.6f} (not trusted)")

    print_action_histogram("\n[Clean Action Histogram]", action_effect["clean_action_histogram"], top_hist_n)
    print_action_histogram("[Perturbed Action Histogram]", action_effect["adv_action_histogram"], top_hist_n)

    print("\n[Most Affected Samples]")
    for row in action_effect["top_examples"]:
        print(
            f"  sample={row['index']:>4d} flip={str(row['flip']).lower():<5s} "
            f"linf={row['linf']:.4f} l2={row['l2']:.4f} "
            f"kl={row['kl_clean_to_adv']:.4f} tv={row['tv_distance']:.4f}"
        )
        print(f"    clean: {row['clean_action_desc']}")
        print(f"    adv  : {row['adv_action_desc']}")

    clean_reward = rollout_clean["mean_reward"]
    attack_reward = rollout_attack["mean_reward"]
    denom = abs(clean_reward) if abs(clean_reward) > 1e-8 else 1e-8
    reward_drop_pct = 100.0 * (clean_reward - attack_reward) / denom

    print("\n[Rollout Reward]")
    print(
        f"  clean mean/std reward     : {rollout_clean['mean_reward']:.6f} / "
        f"{rollout_clean['std_reward']:.6f}"
    )
    print(
        f"  attack mean/std reward    : {rollout_attack['mean_reward']:.6f} / "
        f"{rollout_attack['std_reward']:.6f}"
    )
    print(f"  reward drop               : {clean_reward - attack_reward:.6f} ({reward_drop_pct:+.2f}%)")
    print(f"  rollout attack flip rate  : {rollout_attack['action_flip_rate']:.4f}")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate how pert.h5 changes the saved PPO actor/action net."
    )
    parser.add_argument("--perturbator_path", default="pert.h5")
    parser.add_argument("--policy_dir", default=None)
    parser.add_argument("--collect_steps", type=int, default=200)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--eval_max_steps", type=int, default=10)
    parser.add_argument("--action_mode", choices=["sample", "greedy"], default="greedy")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dense_verify_tol", type=float, default=0.1)
    parser.add_argument("--top_hist_n", type=int, default=6)
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    set_all_seeds(args.seed)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ["ADVORAN_CONFIG_MODULE"] = "config_em_filtered"

    env = ran_env_wrapper.get_eval_env(config_obj=cfg)
    maybe_seed_env(env, args.seed)

    _, actor_net, actor_path = load_actor(env, args.policy_dir)
    perturbator = load_perturbator(args.perturbator_path)
    dense_layers = extract_dense_layers(actor_net)

    print(f"Collected {len(dense_layers)} Dense layers from actor net")
    for index, layer in enumerate(dense_layers):
        print(f"  [{index}] {layer.name} kernel={tuple(layer.kernel.shape)}")
    obs_batch, step_type_batch = collect_rollout_batch(
        env=env,
        actor_net=actor_net,
        num_steps=args.collect_steps,
        action_mode=args.action_mode,
        seed=args.seed,
    )

    action_effect = analyse_action_net_effect(
        actor_net=actor_net,
        perturbator=perturbator,
        obs_batch=obs_batch,
        step_type_batch=step_type_batch,
        collection_action_mode=args.action_mode,
        dense_layers=dense_layers,
        dense_verify_tol=args.dense_verify_tol,
    )

    rollout_clean = evaluate_rollout_reward(
        env=env,
        actor_net=actor_net,
        perturbator=None,
        episodes=args.eval_episodes,
        max_steps=args.eval_max_steps,
        action_mode=args.action_mode,
        seed=args.seed,
    )
    rollout_attack = evaluate_rollout_reward(
        env=env,
        actor_net=actor_net,
        perturbator=perturbator,
        episodes=args.eval_episodes,
        max_steps=args.eval_max_steps,
        action_mode=args.action_mode,
        seed=args.seed,
    )

    summary = {
        "policy_dir": resolve_policy_dir(args.policy_dir),
        "actor_snapshot": actor_path,
        "perturbator_path": args.perturbator_path,
        "seed": args.seed,
        "action_effect": action_effect,
        "rollout_clean": rollout_clean,
        "rollout_attack": rollout_attack,
    }

    print_report(summary, top_hist_n=args.top_hist_n)

    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Wrote JSON summary to {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
