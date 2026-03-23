#!/usr/bin/env python3


import argparse
import importlib
import os
import gc
import json

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import numpy as np
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.utils import common

import agent_builder
import ran_env_wrapper


def parse_args():
    p = argparse.ArgumentParser(description="Modular DRL training launcher")
    p.add_argument("--config_module", default="config_em", help="Config module, e.g. config/config_tp2/config_urllc")
    p.add_argument("--algo", default="ppo", choices=["ppo"], help="DRL algorithm")

    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--num_parallel_environments", type=int, default=None)
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--policy_dir", default=None)
    p.add_argument("--log_dir", default=None)
    p.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Restore from latest checkpoint in checkpoint_dir before training",
    )
    p.add_argument(
        "--action_print_top_k",
        type=int,
        default=8,
        help="Top-K most frequent actions to print at each log interval",
    )

    p.add_argument("--cpu_only", action="store_true", help="Force CPU-only at runtime")
    p.add_argument("--render_eval", action="store_true", help="Render eval environment each step")
    return p.parse_args()


def apply_overrides(cfg, args):
    if args.max_train_steps is not None:
        cfg.max_train_steps = int(args.max_train_steps)
    if args.num_parallel_environments is not None:
        cfg.num_parallel_environments = int(args.num_parallel_environments)
    if args.checkpoint_dir:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.policy_dir:
        cfg.policy_dir = args.policy_dir
    if args.log_dir:
        cfg.log_dir = args.log_dir


def compute_avg_return(environment, policy, num_episodes=1, render=False, log_actions=False):
    """Run evaluation episodes, optionally capturing the actions taken."""
    total_return = 0.0
    all_actions = []
    for _ in range(int(num_episodes)):
        time_step = environment.reset()
        episode_return = 0.0
        episode_actions = []

        while not time_step.is_last():
            action_step = policy.action(time_step)
            action_np = action_step.action.numpy()
            episode_actions.append(np.reshape(action_np, -1).tolist())

            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            if render:
                try:
                    environment.pyenv.render()
                except Exception:
                    pass

        total_return += episode_return
        if log_actions:
            all_actions.append(episode_actions)

    avg_return = total_return / max(1, int(num_episodes))
    avg_return = float(avg_return.numpy()[0]) if hasattr(avg_return, "numpy") else float(avg_return)
    if log_actions:
        return avg_return, all_actions
    return avg_return


def export_trainable_state(agent, cfg, tag="final"):
    """Save actor/value/optimizer weights in a checkpoint-free format."""
    actor_net = getattr(agent, "actor_net", getattr(agent, "_actor_net", None))
    value_net = getattr(agent, "value_net", getattr(agent, "_value_net", None))
    optimizer = getattr(agent, "_optimizer", None) or getattr(agent, "optimizer", None)
    step = int(agent.train_step_counter.numpy())

    os.makedirs(cfg.policy_dir, exist_ok=True)
    artifacts = {}

    def _dump_module(module, prefix):
        weights = {}
        names = []
        for idx, var in enumerate(module.variables):
            key = f"{prefix}_{idx}"
            weights[key] = var.numpy()
            names.append({"saved_key": key, "var_name": var.name})
        path = os.path.join(cfg.policy_dir, f"{prefix}.npz")
        np.savez_compressed(path, **weights)
        return path, names

    if actor_net is not None:
        actor_path, actor_names = _dump_module(actor_net, "actor")
        artifacts["actor_weights"] = actor_path
        artifacts["actor_vars"] = actor_names
    if value_net is not None:
        value_path, value_names = _dump_module(value_net, "value")
        artifacts["value_weights"] = value_path
        artifacts["value_vars"] = value_names
    if optimizer is not None:
        opt_path = os.path.join(cfg.policy_dir, "optimizer.npz")
        opt_vars = []
        if hasattr(optimizer, "variables"):
            opt_vars = list(optimizer.variables())
        elif hasattr(optimizer, "weights"):
            opt_vars = list(optimizer.weights)
        np.savez_compressed(opt_path, **{f"w_{i}": v.numpy() for i, v in enumerate(opt_vars)})
        artifacts["optimizer_weights"] = opt_path
        artifacts["optimizer_count"] = len(opt_vars)

    meta = {
        "last_export_tag": tag,
        "train_step": step,
        "config_module": getattr(cfg, "__name__", "unknown"),
        "dataset_path": getattr(cfg, "dataset_path", ""),
        "entropy_regularization": getattr(cfg, "ppo_entropy_regularization", None),
        "artifacts": artifacts,
    }
    meta_path = os.path.join(cfg.policy_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Trainable state saved ({tag}, canonical files) -> {cfg.policy_dir}")


def main(_):
    args = parse_args()
    os.environ["ADVORAN_CONFIG_MODULE"] = args.config_module

    try:
        multiprocessing.enable_interactive_mode()
    except Exception:
        pass

    if args.cpu_only:
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    cfg = importlib.import_module(args.config_module)
    apply_overrides(cfg, args)
    project_root = os.path.dirname(os.path.abspath(__file__))
    if not args.policy_dir:
        cfg.policy_dir = os.path.join(project_root, "saved_policy", "em-max", "em-agent-filtered")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.policy_dir, exist_ok=True)
    os.environ.setdefault("TMPDIR", cfg.checkpoint_dir)

    print(f"--- RUN ID: {cfg.run_id}")
    print(f"--- CONFIG: {args.config_module} | ALGO: {args.algo}")

    train_env = None
    eval_env = None
    try:
        train_env = ran_env_wrapper.get_training_env(config_obj=cfg)
        eval_env = ran_env_wrapper.get_eval_env(config_obj=cfg)

        agent = agent_builder.create_agent(train_env, algo=args.algo, config_obj=cfg)

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=train_env.batch_size,
            max_length=cfg.collect_steps_per_iteration,
        )

        collect_driver = dynamic_step_driver.DynamicStepDriver(
            train_env,
            agent.collect_policy,
            observers=[replay_buffer.add_batch],
            num_steps=cfg.collect_steps_per_iteration,
        )

        agent.train = common.function(agent.train)
        collect_driver.run = common.function(collect_driver.run)

        train_checkpointer = common.Checkpointer(
            ckpt_dir=cfg.checkpoint_dir,
            max_to_keep=5,
            agent=agent,
            policy=agent.policy,
            global_step=agent.train_step_counter,
        )
        if args.resume_from_checkpoint:
            train_checkpointer.initialize_or_restore()
            print(f"Checkpoint restore enabled. Starting from step {int(agent.train_step_counter.numpy())}")
        else:
            print("Fresh training run: checkpoint restore is disabled.")

        train_summary_writer = tf.summary.create_file_writer(
            cfg.log_dir,
            flush_millis=int(getattr(cfg, "summaries_flush_secs", 10)) * 1000,
        )

        while agent.train_step_counter.numpy() < int(cfg.max_train_steps):
            collect_driver.run()

            experience_dataset = replay_buffer.as_dataset(
                sample_batch_size=train_env.batch_size,
                num_steps=cfg.collect_steps_per_iteration,
                single_deterministic_pass=True,
            )
            try:
                trajectories, _ = next(iter(experience_dataset))
            except StopIteration:
                trajectories = replay_buffer.gather_all()
                if tf.size(trajectories.step_type) == 0:
                    replay_buffer.clear()
                    continue

            train_loss = agent.train(experience=trajectories)

            step = int(agent.train_step_counter.numpy())

            if step % int(cfg.log_interval) == 0:
                entropy_loss = float(train_loss.extra.entropy_regularization_loss.numpy())
                total_loss = float(train_loss.loss.numpy())
                print(f"Step {step}: loss={total_loss:.6f}, entropy={entropy_loss:.6f}")
                actions_np = tf.reshape(tf.cast(trajectories.action, tf.int32), [-1]).numpy()
                if actions_np.size > 0:
                    top_k = max(1, int(args.action_print_top_k))
                    uniq, counts = np.unique(actions_np, return_counts=True)
                    order = np.argsort(counts)[::-1]
                    top_rows = [f"{int(uniq[i])}:{int(counts[i])}" for i in order[:top_k]]
                    preview = actions_np[: min(20, actions_np.size)].tolist()
                    print(f"Step {step}: actions_preview={preview}")
                    print(f"Step {step}: actions_top{top_k}={', '.join(top_rows)}")
                with train_summary_writer.as_default():
                    tf.summary.scalar("loss/total_loss", train_loss.loss, step=step)
                    tf.summary.scalar("loss/entropy_loss", train_loss.extra.entropy_regularization_loss, step=step)

            if step % int(cfg.eval_interval) == 0:
                avg_return, eval_actions = compute_avg_return(
                    eval_env,
                    agent.policy,
                    num_episodes=int(cfg.num_eval_episodes),
                    render=bool(args.render_eval),
                    log_actions=True,
                )
                print(f"Step {step}: Avg Return = {avg_return:.6f}")
                print(f"Step {step}: Eval actions (episode-wise): {eval_actions}")
                with train_summary_writer.as_default():
                    tf.summary.scalar("metrics/average_return", avg_return, step=step)

            if step % int(cfg.checkpoint_interval) == 0:
                print(f"Saving checkpoint at step {step}")
                try:
                    train_checkpointer.save(global_step=agent.train_step_counter)
                except Exception as exc:
                    print(f"Checkpoint save failed at step {step}: {exc}")
                export_trainable_state(agent, cfg, tag=f"step{step}")

            replay_buffer.clear()
            del trajectories
            gc.collect()

        print("Saving final trainable snapshot (gradient-restorable)...")
        export_trainable_state(agent, cfg, tag="final")
    finally:
        if train_env is not None:
            try:
                train_env.close()
            except Exception:
                pass
        if eval_env is not None:
            try:
                eval_env.close()
            except Exception:
                pass


if __name__ == "__main__":
    main(None)
