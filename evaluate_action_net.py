#!/usr/bin/env python3

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

import config_em_filtered as cfg
import ran_env_wrapper
from evaluate_perturbator_effect import (
    action_from_dist,
    collect_rollout_batch,
    dense_forward_trace,
    evaluate_rollout_reward,
    extract_dense_layers,
    format_action,
    get_dist,
    get_logits,
    get_probs,
    load_actor,
    maybe_seed_env,
    resolve_policy_dir,
    set_all_seeds,
)


def summarize_layer_activations(
    actor_net,
    obs: tf.Tensor,
    step_type: tf.Tensor,
    dense_layers: List[tf.keras.layers.Dense],
    verify_tol: float,
) -> Tuple[Optional[List[dict]], Optional[float]]:
    if not dense_layers:
        return None, None

    activations = dense_forward_trace(dense_layers, obs)
    logits = get_logits(get_dist(actor_net, obs, step_type, training=False))
    final_logits = activations[-1]
    mae = float(tf.reduce_mean(tf.abs(final_logits - logits)).numpy())
    if mae > verify_tol:
        print(
            f"[layer-trace] skipped: dense-trace final-logit MAE={mae:.4f} "
            f"(tol={verify_tol:.4f})"
        )
        return None, mae

    rows = []
    for idx, (layer, act) in enumerate(zip(dense_layers, activations)):
        abs_act = tf.abs(act)
        rows.append(
            {
                "index": idx,
                "name": layer.name,
                "units": int(act.shape[-1]),
                "mean_abs_activation": float(tf.reduce_mean(abs_act).numpy()),
                "std_activation": float(tf.math.reduce_std(act).numpy()),
                "max_abs_activation": float(tf.reduce_max(abs_act).numpy()),
            }
        )
    return rows, mae


def _sorted_probability_stats(probs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    sorted_probs = tf.sort(probs, direction="DESCENDING", axis=-1)
    top_prob = sorted_probs[:, 0]
    if int(sorted_probs.shape[-1]) > 1:
        top_margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    else:
        top_margin = sorted_probs[:, 0]
    return top_prob, top_margin


def analyse_policy_action_net(
    actor_net,
    obs_batch: np.ndarray,
    step_type_batch: np.ndarray,
    analysis_action_mode: str,
    dense_layers: List[tf.keras.layers.Dense],
    dense_verify_tol: float,
    top_examples: int,
) -> Dict:
    obs = tf.convert_to_tensor(obs_batch, dtype=tf.float32)
    step_type = tf.reshape(tf.cast(step_type_batch, tf.int32), [-1])

    dist = get_dist(actor_net, obs, step_type, training=False)
    probs = get_probs(dist)
    logits = get_logits(dist)
    actions = action_from_dist(dist, analysis_action_mode)

    entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=-1)
    top_prob, top_margin = _sorted_probability_stats(probs)
    logits_l2 = tf.norm(logits, ord=2, axis=-1)

    layer_rows, dense_trace_mae = summarize_layer_activations(
        actor_net=actor_net,
        obs=obs,
        step_type=step_type,
        dense_layers=dense_layers,
        verify_tol=dense_verify_tol,
    )

    action_hist = np.bincount(actions.numpy(), minlength=cfg.n_actions)
    unique_actions = int(np.count_nonzero(action_hist))

    entropy_np = entropy.numpy()
    top_prob_np = top_prob.numpy()
    top_margin_np = top_margin.numpy()
    logits_l2_np = logits_l2.numpy()
    actions_np = actions.numpy()

    high_conf_idx = np.argsort(top_prob_np)[::-1][: min(top_examples, obs_batch.shape[0])]
    low_margin_idx = np.argsort(top_margin_np)[: min(top_examples, obs_batch.shape[0])]

    return {
        "num_samples": int(obs_batch.shape[0]),
        "obs_dim": int(obs_batch.shape[1]),
        "analysis_action_mode": analysis_action_mode,
        "entropy_mean": float(np.mean(entropy_np)),
        "entropy_std": float(np.std(entropy_np)),
        "top_prob_mean": float(np.mean(top_prob_np)),
        "top_prob_max": float(np.max(top_prob_np)),
        "top_margin_mean": float(np.mean(top_margin_np)),
        "top_margin_min": float(np.min(top_margin_np)),
        "logit_l2_mean": float(np.mean(logits_l2_np)),
        "unique_actions": unique_actions,
        "action_histogram": action_hist.astype(int).tolist(),
        "dense_trace_mae": dense_trace_mae,
        "layer_summaries": layer_rows,
        "most_confident_samples": [
            {
                "index": int(idx),
                "action": int(actions_np[idx]),
                "action_desc": format_action(int(actions_np[idx])),
                "top_prob": float(top_prob_np[idx]),
                "top_margin": float(top_margin_np[idx]),
                "entropy": float(entropy_np[idx]),
                "logit_l2": float(logits_l2_np[idx]),
            }
            for idx in high_conf_idx
        ],
        "lowest_margin_samples": [
            {
                "index": int(idx),
                "action": int(actions_np[idx]),
                "action_desc": format_action(int(actions_np[idx])),
                "top_prob": float(top_prob_np[idx]),
                "top_margin": float(top_margin_np[idx]),
                "entropy": float(entropy_np[idx]),
                "logit_l2": float(logits_l2_np[idx]),
            }
            for idx in low_margin_idx
        ],
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


def print_example_block(title: str, rows: List[dict]) -> None:
    print(title)
    if not rows:
        print("  none")
        return
    for row in rows:
        print(
            f"  sample={row['index']:>4d} "
            f"top_prob={row['top_prob']:.4f} "
            f"margin={row['top_margin']:.4f} "
            f"entropy={row['entropy']:.4f} "
            f"logit_l2={row['logit_l2']:.4f}"
        )
        print(f"    {row['action_desc']}")


def print_report(summary: Dict, top_hist_n: int) -> None:
    analysis = summary["analysis"]
    rollout = summary["rollout"]

    print("\n" + "=" * 72)
    print("Saved Policy Action Net Evaluation")
    print("=" * 72)
    print(
        f"policy_dir     : {summary['policy_dir']}\n"
        f"actor_snapshot : {summary['actor_snapshot']}\n"
        f"samples        : {analysis['num_samples']}  "
        f"obs_dim={analysis['obs_dim']}  "
        f"analysis_mode={analysis['analysis_action_mode']}"
    )

    print("\n[Action Net Summary]")
    print(f"  unique actions used        : {analysis['unique_actions']}/{cfg.n_actions}")
    print(
        f"  entropy mean/std          : {analysis['entropy_mean']:.6f} / "
        f"{analysis['entropy_std']:.6f}"
    )
    print(
        f"  top prob mean/max         : {analysis['top_prob_mean']:.6f} / "
        f"{analysis['top_prob_max']:.6f}"
    )
    print(
        f"  top margin mean/min       : {analysis['top_margin_mean']:.6f} / "
        f"{analysis['top_margin_min']:.6f}"
    )
    print(f"  logit L2 mean             : {analysis['logit_l2_mean']:.6f}")

    if analysis["layer_summaries"] is not None:
        print(f"  dense-trace MAE vs logits : {analysis['dense_trace_mae']:.6f}")
        print("  layer activations:")
        for row in analysis["layer_summaries"]:
            print(
                f"    [{row['index']}] {row['name']:<24s} "
                f"mean_abs={row['mean_abs_activation']:.6f} "
                f"std={row['std_activation']:.6f} "
                f"max_abs={row['max_abs_activation']:.6f}"
            )
    elif analysis["dense_trace_mae"] is not None:
        print(f"  dense-trace MAE vs logits : {analysis['dense_trace_mae']:.6f} (not trusted)")

    print_action_histogram("\n[Action Histogram]", analysis["action_histogram"], top_hist_n)
    print_example_block("\n[Most Confident Samples]", analysis["most_confident_samples"])
    print_example_block("[Lowest Margin Samples]", analysis["lowest_margin_samples"])

    print("\n[Rollout Reward]")
    print(
        f"  mean/std reward           : {rollout['mean_reward']:.6f} / "
        f"{rollout['std_reward']:.6f}"
    )
    print(f"  mean episode length       : {rollout['mean_ep_len']:.4f}")
    print("=" * 72)


def evaluate_policy(
    env,
    policy_dir: Optional[str],
    collect_steps: int,
    eval_episodes: int,
    eval_max_steps: int,
    collection_action_mode: str,
    analysis_action_mode: str,
    dense_verify_tol: float,
    top_examples: int,
    seed: int,
) -> Dict:
    _, actor_net, actor_path = load_actor(env, policy_dir)
    dense_layers = extract_dense_layers(actor_net)

    print(f"Collected {len(dense_layers)} Dense layers from actor net")
    for index, layer in enumerate(dense_layers):
        print(f"  [{index}] {layer.name} kernel={tuple(layer.kernel.shape)}")
    obs_batch, step_type_batch = collect_rollout_batch(
        env=env,
        actor_net=actor_net,
        num_steps=collect_steps,
        action_mode=collection_action_mode,
        seed=seed,
    )
    analysis = analyse_policy_action_net(
        actor_net=actor_net,
        obs_batch=obs_batch,
        step_type_batch=step_type_batch,
        analysis_action_mode=analysis_action_mode,
        dense_layers=dense_layers,
        dense_verify_tol=dense_verify_tol,
        top_examples=top_examples,
    )
    rollout = evaluate_rollout_reward(
        env=env,
        actor_net=actor_net,
        perturbator=None,
        episodes=eval_episodes,
        max_steps=eval_max_steps,
        action_mode=analysis_action_mode,
        seed=seed,
    )

    return {
        "policy_dir": resolve_policy_dir(policy_dir),
        "actor_snapshot": actor_path,
        "seed": seed,
        "collection_action_mode": collection_action_mode,
        "analysis": analysis,
        "rollout": rollout,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the actor/action net of one or more saved policies."
    )
    parser.add_argument(
        "policy_dirs",
        nargs="*",
        help="Saved policy directories. If omitted, use the repo default policy snapshot.",
    )
    parser.add_argument("--collect_steps", type=int, default=200)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--eval_max_steps", type=int, default=10)
    parser.add_argument("--collection_action_mode", choices=["sample", "greedy"], default="greedy")
    parser.add_argument("--analysis_action_mode", choices=["sample", "greedy"], default="greedy")
    parser.add_argument("--dense_verify_tol", type=float, default=0.1)
    parser.add_argument("--top_examples", type=int, default=5)
    parser.add_argument("--top_hist_n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    set_all_seeds(args.seed)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ["ADVORAN_CONFIG_MODULE"] = "config_em_filtered"

    env = ran_env_wrapper.get_eval_env(config_obj=cfg)
    maybe_seed_env(env, args.seed)

    policy_dirs = args.policy_dirs or [None]
    results = []
    for policy_dir in policy_dirs:
        summary = evaluate_policy(
            env=env,
            policy_dir=policy_dir,
            collect_steps=args.collect_steps,
            eval_episodes=args.eval_episodes,
            eval_max_steps=args.eval_max_steps,
            collection_action_mode=args.collection_action_mode,
            analysis_action_mode=args.analysis_action_mode,
            dense_verify_tol=args.dense_verify_tol,
            top_examples=args.top_examples,
            seed=args.seed,
        )
        print_report(summary, top_hist_n=args.top_hist_n)
        results.append(summary)

    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = results[0] if len(results) == 1 else results
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote JSON summary to {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
