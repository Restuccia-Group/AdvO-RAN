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
import ran_env_wrapper
import config_em_filtered as cfg


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_step_type_shape(step_type: tf.Tensor) -> tf.Tensor:
    return tf.reshape(tf.cast(step_type, tf.int32), [-1])


def ensure_action_shape(actions: tf.Tensor) -> tf.Tensor:
    return tf.reshape(tf.cast(actions, tf.int32), [-1])


DEFAULT_POLICY_DIR_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "saved_policy", "em-max", "em-agent-filtered"),
    os.path.join(os.path.dirname(__file__), "saved_policy", "em-agent-filtered"),
    os.path.join(os.path.dirname(__file__), "saved_policy", "em-agent"),
]


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
        looked_for = ", ".join(os.path.basename(p) for p in candidates)
        raise FileNotFoundError(
            f"Could not find {prefix} snapshot in {policy_dir}. Tried: {looked_for}"
        )
    return candidates[0]


def load_snapshot_paths(policy_dir: Optional[str] = None) -> Dict[str, str]:
    policy_dir = resolve_policy_dir(policy_dir)
    return {
        "actor":     resolve_snapshot_file(policy_dir, "actor", "npz", required=True),
        "value":     resolve_snapshot_file(policy_dir, "value", "npz", required=False),
        "optimizer": resolve_snapshot_file(policy_dir, "optimizer", "npz", required=False),
        "meta":      resolve_snapshot_file(policy_dir, "metadata", "json", required=False),
    }


def read_json_if_exists(path: Optional[str]) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Warning: could not read metadata {path}: {exc}")
        return None


def find_variable_map(meta: Optional[dict], target: str = "actor") -> Optional[List[dict]]:
    if meta is None:
        return None
    candidate_keys = [
        f"{target}_variable_map",
        f"{target}_var_map",
        f"{target}_map",
        f"{target}_vars",
    ]
    for key in candidate_keys:
        if key in meta and isinstance(meta[key], list):
            return meta[key]
    for _, value in meta.items():
        if isinstance(value, dict):
            for key in candidate_keys:
                if key in value and isinstance(value[key], list):
                    return value[key]
    return None


def load_npz_to_vars(
    npz_path: str,
    variables: List[tf.Variable],
    variable_map: Optional[List[dict]] = None,
) -> None:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Snapshot not found: {npz_path}")
    data = np.load(npz_path)
    if variable_map:
        name_to_var = {v.name: v for v in variables}
        loaded = 0
        for item in variable_map:
            saved_key = item.get("saved_key")
            var_name  = item.get("var_name")
            if saved_key not in data or var_name not in name_to_var:
                continue
            arr = data[saved_key]
            var = name_to_var[var_name]
            if tuple(var.shape) != tuple(arr.shape):
                print(f"Skip shape mismatch {var_name}: var={tuple(var.shape)} file={tuple(arr.shape)}")
                continue
            var.assign(arr)
            loaded += 1
        if loaded > 0:
            print(f"Loaded {loaded}/{len(variables)} vars from {npz_path} via metadata map")
            return
        print("Metadata map matched no variables – falling back to raw order.")
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


def save_vars_to_npz(npz_path: str, variables: List[tf.Variable], prefix: str = "actor") -> List[dict]:
    arrays = {}
    variable_map = []
    for i, var in enumerate(variables):
        key = f"{prefix}_{i}"
        arrays[key] = var.numpy()
        variable_map.append({"saved_key": key, "var_name": var.name})
    np.savez(npz_path, **arrays)
    return variable_map


def get_actor_net(agent):
    return getattr(agent, "actor_net", getattr(agent, "_actor_net"))


def get_rollout_policy(agent):
    return getattr(agent, "collect_policy", getattr(agent, "policy"))


def build_actor_once(env, actor_net) -> None:
    ts        = env.reset()
    obs       = tf.convert_to_tensor(ts.observation, dtype=tf.float32)
    step_type = ensure_step_type_shape(tf.convert_to_tensor(ts.step_type, dtype=tf.int32))
    batch_size= tf.shape(obs)[0]
    init_state= actor_net.get_initial_state(batch_size=batch_size)
    _         = actor_net(obs, step_type, init_state, training=False)


def get_dist(actor_net, obs: tf.Tensor, step_type: tf.Tensor, training: bool = False):
    step_type  = ensure_step_type_shape(step_type)
    batch_size = tf.shape(obs)[0]
    init_state = actor_net.get_initial_state(batch_size=batch_size)
    dist, _    = actor_net(obs, step_type, init_state, training=training)
    return dist


def greedy_action_from_dist(dist) -> tf.Tensor:
    try:
        action = dist.mode()
    except Exception:
        try:
            action = tf.argmax(dist.logits_parameter(), axis=-1, output_type=tf.int32)
        except Exception:
            action = tf.argmax(dist.probs_parameter(), axis=-1, output_type=tf.int32)
    return ensure_action_shape(action)


def sample_action_from_dist(dist) -> tf.Tensor:
    try:
        action = dist.sample()
    except Exception:
        if hasattr(dist, "logits_parameter"):
            logits = tf.convert_to_tensor(dist.logits_parameter(), dtype=tf.float32)
        else:
            probs = tf.convert_to_tensor(dist.probs_parameter(), dtype=tf.float32)
            logits = tf.math.log(probs + 1e-8)
        if logits.shape.rank == 3 and logits.shape[1] == 1:
            logits = tf.squeeze(logits, axis=1)
        action = tf.random.categorical(logits, 1, dtype=tf.int32)
    return ensure_action_shape(action)


def maybe_seed_env(env, seed: int) -> None:
    try:
        if hasattr(env, "seed"):
            env.seed(seed)
    except Exception:
        pass


class PerturbationNet(tf.keras.Model):
    def __init__(self, obs_dim, eps, hidden=(256, 256), obs_mean=None, obs_std=None):
        super().__init__(name="perturbation_net")
        self.eps = eps
        self.obs_mean = tf.constant(
            obs_mean if obs_mean is not None else np.zeros(obs_dim),
            dtype=tf.float32,
        )
        self.obs_std = tf.constant(
            obs_std if obs_std is not None else np.ones(obs_dim),
            dtype=tf.float32,
        )
        self.hidden_layers = [
            tf.keras.layers.Dense(h, activation="relu", name=f"hidden_{i}")
            for i, h in enumerate(hidden)
        ]
        self.out_layer = tf.keras.layers.Dense(obs_dim, name="delta_out")

    def call(self, obs, training=False):
        x = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return tf.clip_by_value(self.out_layer(x), -self.eps, self.eps)


_perturbator_model: Optional[tf.keras.Model] = None


def load_perturbator(path: str) -> tf.keras.Model:

    global _perturbator_model
    if _perturbator_model is not None:
        return _perturbator_model

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Perturbator model not found: {path}\n"
            "Pass the correct path via --perturbator_path."
        )

    print(f"\n[PERTURBATOR] Loading model from: {path}")
    _perturbator_model = tf.keras.models.load_model(path, compile=False)
    def _maybe_shape(name, fn):
        try:
            value = fn()
        except AttributeError:
            print(f"[PERTURBATOR] {name} shape unavailable for subclassed models.")
        else:
            print(f"[PERTURBATOR] {name} shape : {value}")

    _maybe_shape("Input", lambda: _perturbator_model.input_shape)
    _maybe_shape("Output", lambda: _perturbator_model.output_shape)
    return _perturbator_model


def perturb_obs(perturbator: tf.keras.Model, obs: tf.Tensor) -> tf.Tensor:

    obs = tf.cast(obs, tf.float32)
    delta = tf.cast(perturbator(obs, training=False), tf.float32)

    if tuple(delta.shape) != tuple(obs.shape):
        raise RuntimeError(
            f"Perturbator output shape {tuple(delta.shape)} does not match "
            f"observation shape {tuple(obs.shape)}.  "
            "The model must output a tensor of the same shape as its input."
        )

    # The learned perturbator is trained to output delta, not the full adversarial observation.
    return tf.stop_gradient(obs + delta)


def perturbator_attack_obs(
    perturbator: tf.keras.Model,
    actor_net,
    obs: tf.Tensor,
    step_type: tf.Tensor,
    action_mode: str = "greedy",
) -> Dict[str, tf.Tensor]:

    obs       = tf.cast(obs, tf.float32)
    step_type = ensure_step_type_shape(step_type)
    action_fn = sample_action_from_dist if action_mode == "sample" else greedy_action_from_dist

    clean_dist   = get_dist(actor_net, obs, step_type, training=False)
    clean_action = action_fn(clean_dist)

    adv_obs  = perturb_obs(perturbator, obs)

    adv_dist   = get_dist(actor_net, adv_obs, step_type, training=False)
    adv_action = action_fn(adv_dist)

    return {
        "adv_obs":      adv_obs,
        "clean_action": clean_action,
        "adv_action":   adv_action,
    }


def _activation_name(act) -> str:
    """Normalise a Keras activation to a lowercase string."""
    if act is None:
        return "linear"
    if hasattr(act, "__name__"):
        return act.__name__.lower()
    if hasattr(act, "get_config"):
        try:
            return act.get_config().get("name", "").lower()
        except Exception:
            pass
    return str(act).lower()


def _collect_dense_layers_dfs(module) -> List[tf.keras.layers.Dense]:

    collected: List[tf.keras.layers.Dense] = []
    seen_ids = set()

    def _visit(obj):
        oid = id(obj)
        if oid in seen_ids:
            return
        seen_ids.add(oid)

        if isinstance(obj, tf.keras.layers.Dense):
            collected.append(obj)
            return

        for attr in ("_self_tracked_trackables", "layers", "_layers", "_sublayers"):
            sublayers = getattr(obj, attr, None)
            if sublayers is None:
                continue
            try:
                it = iter(sublayers)
            except TypeError:
                continue
            for sub in it:
                _visit(sub)
            break

    _visit(module)
    return collected


def extract_dense_layers(actor_net) -> List[tf.keras.layers.Dense]:

    for sub_attr in ("_encoding_network", "encoding_network"):
        enc = getattr(actor_net, sub_attr, None)
        if enc is not None:
            layers = _collect_dense_layers_dfs(enc)
            for proj_attr in ("_output_layers", "_projection_networks",
                               "_action_projection_layer", "_output_tensor_spec"):
                proj = getattr(actor_net, proj_attr, None)
                if proj is not None:
                    layers += _collect_dense_layers_dfs(proj)
            if layers:
                return layers

    return _collect_dense_layers_dfs(actor_net)


def ibp_dense(
    W: tf.Tensor, b: tf.Tensor,
    x_lo: tf.Tensor, x_hi: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    W_pos = tf.maximum(W, 0.0)
    W_neg = tf.minimum(W, 0.0)
    y_lo  = x_lo @ W_pos + x_hi @ W_neg + b
    y_hi  = x_hi @ W_pos + x_lo @ W_neg + b
    return y_lo, y_hi


def ibp_relu(x_lo: tf.Tensor, x_hi: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.maximum(x_lo, 0.0), tf.maximum(x_hi, 0.0)


def ibp_tanh(x_lo: tf.Tensor, x_hi: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.tanh(x_lo), tf.tanh(x_hi)


def ibp_sigmoid(x_lo: tf.Tensor, x_hi: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.sigmoid(x_lo), tf.sigmoid(x_hi)


def ibp_activation(
    act_name: str,
    x_lo: tf.Tensor,
    x_hi: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    if "relu" in act_name:
        return ibp_relu(x_lo, x_hi)
    elif "tanh" in act_name:
        return ibp_tanh(x_lo, x_hi)
    elif "sigmoid" in act_name:
        return ibp_sigmoid(x_lo, x_hi)
    else:
        return x_lo, x_hi


def ibp_actor_forward(
    dense_layers: List[tf.keras.layers.Dense],
    obs: tf.Tensor,
    eps: float,
) -> Tuple[tf.Tensor, tf.Tensor]:
    x_lo = obs - eps
    x_hi = obs + eps

    for layer in dense_layers:
        W    = layer.kernel
        b    = layer.bias
        x_lo, x_hi = ibp_dense(W, b, x_lo, x_hi)

        act_name = _activation_name(layer.activation)
        x_lo, x_hi = ibp_activation(act_name, x_lo, x_hi)

    return x_lo, x_hi


def verify_ibp_accuracy(
    actor_net,
    dense_layers: List[tf.keras.layers.Dense],
    obs: tf.Tensor,
    step_type: tf.Tensor,
    tol: float = 0.1,
) -> float:

    logit_lo, logit_hi = ibp_actor_forward(dense_layers, obs, eps=0.0)
    ibp_centre = (logit_lo + logit_hi) / 2.0

    dist = get_dist(actor_net, obs, step_type, training=False)
    try:
        true_logits = tf.cast(dist.logits_parameter(), tf.float32)
    except AttributeError:
        true_logits = tf.math.log(tf.cast(dist.probs_parameter(), tf.float32) + 1e-9)

    mae = float(tf.reduce_mean(tf.abs(ibp_centre - true_logits)).numpy())
    if mae > tol:
        print(
            f"[IBP VERIFY] WARNING: IBP centre MAE vs true logits = {mae:.4f} "
            f"(> tol {tol}).  Layer ordering may be incorrect. "
            f"RADIAL loss is still computed but the certified guarantee is weakened."
        )
    else:
        print(f"[IBP VERIFY] OK — centre MAE = {mae:.4f}")
    return mae


def radial_kl_loss(
    actor_net,
    dense_layers,
    obs,
    step_type,
    eps: float,
    training: bool = True,
):

    obs       = tf.cast(obs, tf.float32)
    step_type = ensure_step_type_shape(step_type)

    clean_dist = get_dist(actor_net, obs, step_type, training=training)
    try:
        clean_logits = tf.cast(clean_dist.logits_parameter(), tf.float32)
    except AttributeError:
        clean_logits = tf.math.log(
            tf.cast(clean_dist.probs_parameter(), tf.float32) + 1e-9
        )

    num_actions     = tf.cast(tf.shape(clean_logits)[-1], tf.float32)
    clean_log_probs = tf.nn.log_softmax(clean_logits)
    clean_probs     = tf.nn.softmax(clean_logits)

    logit_lo, logit_hi = ibp_actor_forward(dense_layers, obs, eps)

    logZ_upper   = tf.reduce_logsumexp(logit_hi, axis=-1, keepdims=True)
    lb_log_probs = logit_lo - logZ_upper

    kl_per_sample = tf.reduce_sum(
        clean_probs * (clean_log_probs - lb_log_probs), axis=-1
    )
    kl_per_sample = tf.maximum(kl_per_sample, 0.0)
    kl_raw        = tf.reduce_mean(kl_per_sample)

    log_num_actions = tf.math.log(tf.maximum(num_actions, 2.0))
    kl_normalised   = kl_raw / log_num_actions

    return kl_normalised, kl_raw


def curriculum_eps(
    iteration: int,
    eps_final: float,
    eps_start: float = 0.0,
    schedule_start: int = 0,
    schedule_end: int = 500,
) -> float:
    if iteration < schedule_start:
        return eps_start
    if iteration >= schedule_end:
        return eps_final
    frac = (iteration - schedule_start) / max(schedule_end - schedule_start, 1)
    return float(eps_start + frac * (eps_final - eps_start))


def adaptive_radial_weight(
    base_weight: float,
    reward_history: List[float],
    window: int = 20,
    drop_tol: float = 0.10,
    recover_rate: float = 0.02,
    floor: float = 0.01,
    state: Optional[Dict] = None,
) -> Tuple[float, Dict]:
    if state is None:
        state = {"current_weight": base_weight}

    w = state["current_weight"]

    if len(reward_history) < 2 * window:
        state["current_weight"] = base_weight
        return base_weight, state

    baseline = float(np.mean(reward_history[-2 * window : -window]))
    recent   = float(np.mean(reward_history[-window:]))

    if baseline == 0.0:
        state["current_weight"] = w
        return w, state

    relative_drop = (baseline - recent) / abs(baseline)

    if relative_drop > drop_tol:
        w = max(w * 0.5, floor)
        print(
            f"  [RADIAL-ADAPT] reward dropped {relative_drop:.1%} "
            f"(baseline={baseline:.4f} recent={recent:.4f}) "
            f"-> radial_weight halved to {w:.4f}"
        )
    else:
        w = min(w + recover_rate * base_weight, base_weight)

    state["current_weight"] = w
    return w, state


def discounted_returns(rewards, gamma: float = 0.99) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32)
    out     = np.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        out[t]  = running
    return out


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean()) / (x.std() + 1e-8)


def collect_rollout(env, agent, actor_net, horizon: int, gamma: float) -> Dict[str, np.ndarray]:
    obs_buf, step_type_buf, action_buf, reward_buf, logp_buf = [], [], [], [], []
    policy = get_rollout_policy(agent)
    ts     = env.reset()

    for _ in range(horizon):
        obs       = tf.convert_to_tensor(ts.observation, dtype=tf.float32)
        step_type = ensure_step_type_shape(tf.convert_to_tensor(ts.step_type, dtype=tf.int32))

        action_step = policy.action(ts)
        action      = ensure_action_shape(action_step.action)

        dist     = get_dist(actor_net, obs, step_type, training=False)
        log_prob = tf.reshape(tf.cast(dist.log_prob(action), tf.float32), [-1])

        next_ts = env.step(tf.reshape(action, [-1]))

        obs_buf.append(obs.numpy()[0])
        step_type_buf.append(int(step_type.numpy().reshape(-1)[0]))
        action_buf.append(int(action.numpy().reshape(-1)[0]))
        reward_buf.append(float(np.asarray(next_ts.reward.numpy()).reshape(-1)[0]))
        logp_buf.append(float(log_prob.numpy().reshape(-1)[0]))

        ts = next_ts
        if ts.is_last():
            ts = env.reset()

    returns    = discounted_returns(reward_buf, gamma=gamma)
    advantages = normalize(returns)

    return {
        "obs":           np.asarray(obs_buf,       dtype=np.float32),
        "step_type":     np.asarray(step_type_buf, dtype=np.int32),
        "actions":       np.asarray(action_buf,    dtype=np.int32),
        "rewards":       np.asarray(reward_buf,    dtype=np.float32),
        "returns":       np.asarray(returns,       dtype=np.float32),
        "advantages":    np.asarray(advantages,    dtype=np.float32),
        "old_log_probs": np.asarray(logp_buf,      dtype=np.float32),
    }


def ppo_actor_loss(
    actor_net,
    obs: tf.Tensor,
    step_type: tf.Tensor,
    actions: tf.Tensor,
    advantages: tf.Tensor,
    old_log_probs: tf.Tensor,
    clip_ratio: float   = 0.2,
    entropy_coef: float = 0.01,
    training: bool      = False,
):
    actions       = ensure_action_shape(actions)
    step_type     = ensure_step_type_shape(step_type)
    advantages    = tf.reshape(tf.cast(advantages,    tf.float32), [-1])
    old_log_probs = tf.reshape(tf.cast(old_log_probs, tf.float32), [-1])

    dist          = get_dist(actor_net, obs, step_type, training=training)
    new_log_probs = tf.reshape(tf.cast(dist.log_prob(actions), tf.float32), [-1])

    ratio  = tf.exp(new_log_probs - old_log_probs)
    surr1  = ratio * advantages
    surr2  = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages

    policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
    entropy     = tf.reduce_mean(tf.reshape(tf.cast(dist.entropy(), tf.float32), [-1]))
    total_loss  = policy_loss - entropy_coef * entropy

    return total_loss, policy_loss, entropy


def train_actor_step_radial(
    actor_net,
    dense_layers,
    optimizer,
    obs,
    step_type,
    actions,
    advantages,
    old_log_probs,
    clip_ratio: float,
    entropy_coef: float,
    clean_weight: float,
    radial_weight: float,
    current_eps: float,
    max_grad_norm: float,
    max_kl_clip: float = 1.0,
) -> Dict[str, float]:
    obs           = tf.cast(obs, tf.float32)
    step_type     = ensure_step_type_shape(step_type)
    actions       = ensure_action_shape(actions)
    advantages    = tf.reshape(tf.cast(advantages,    tf.float32), [-1])
    old_log_probs = tf.reshape(tf.cast(old_log_probs, tf.float32), [-1])

    with tf.GradientTape() as tape:
        ppo_loss, policy_loss, entropy = ppo_actor_loss(
            actor_net     = actor_net,
            obs           = obs,
            step_type     = step_type,
            actions       = actions,
            advantages    = advantages,
            old_log_probs = old_log_probs,
            clip_ratio    = clip_ratio,
            entropy_coef  = entropy_coef,
            training      = True,
        )

        if radial_weight > 0.0 and current_eps > 0.0:
            kl_normalised, kl_raw = radial_kl_loss(
                actor_net    = actor_net,
                dense_layers = dense_layers,
                obs          = obs,
                step_type    = step_type,
                eps          = current_eps,
                training     = True,
            )
            kl_clipped = tf.minimum(kl_normalised, max_kl_clip)
        else:
            kl_normalised = tf.constant(0.0, dtype=tf.float32)
            kl_raw        = tf.constant(0.0, dtype=tf.float32)
            kl_clipped    = tf.constant(0.0, dtype=tf.float32)

        total_loss = clean_weight * ppo_loss + radial_weight * kl_clipped

    grads = tape.gradient(total_loss, actor_net.trainable_variables)
    grads = [g if g is not None else tf.zeros_like(v)
             for g, v in zip(grads, actor_net.trainable_variables)]
    grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
    optimizer.apply_gradients(zip(grads, actor_net.trainable_variables))

    return {
        "loss":        float(total_loss.numpy()),
        "ppo_loss":    float(ppo_loss.numpy()),
        "policy_loss": float(policy_loss.numpy()),
        "kl_norm":     float(kl_normalised.numpy()),
        "kl_raw":      float(kl_raw.numpy()),
        "kl_clipped":  float(kl_clipped.numpy()),
        "entropy":     float(entropy.numpy()),
        "grad_norm":   float(grad_norm.numpy()),
        "current_eps": current_eps,
    }


def evaluate_actor(
    perturbator: tf.keras.Model,
    actor_net,
    env,
    episodes: int,
    max_steps: int,
    attack: bool,
    action_mode: str = "sample",
    log_actions: bool = False,
) -> Dict[str, float]:
    """
    Evaluate the actor over `episodes` episodes.

    When attack=True the perturbator model is used to perturb each
    observation before the actor selects an action.  No PGD parameters
    are needed; the perturbator handles adversarial generation entirely.

    `action_mode="sample"` matches the perturbator trainer, which
    evaluates through the policy sampler rather than argmax.
    """
    if action_mode not in {"sample", "greedy"}:
        raise ValueError(f"Unsupported eval action mode: {action_mode}")

    action_fn = sample_action_from_dist if action_mode == "sample" else greedy_action_from_dist
    episode_rewards, episode_lengths = [], []
    flip_count  = 0
    total_steps = 0
    action_ids: List[int] = []

    for _ in range(episodes):
        ts        = env.reset()
        ep_reward = 0.0
        ep_len    = 0

        while (not ts.is_last()) and ep_len < max_steps:
            obs       = tf.convert_to_tensor(ts.observation, dtype=tf.float32)
            step_type = ensure_step_type_shape(
                tf.convert_to_tensor(ts.step_type, dtype=tf.int32)
            )

            clean_dist   = get_dist(actor_net, obs, step_type, training=False)
            clean_action = action_fn(clean_dist)

            if attack:
                atk    = perturbator_attack_obs(
                    perturbator, actor_net, obs, step_type, action_mode=action_mode
                )
                action = atk["adv_action"]
                if (int(action.numpy().reshape(-1)[0]) !=
                        int(clean_action.numpy().reshape(-1)[0])):
                    flip_count += 1
            else:
                action = clean_action

            action_id = int(action.numpy().reshape(-1)[0])
            action_ids.append(action_id)
            ts          = env.step(tf.reshape(action, [-1]))
            reward      = float(np.asarray(ts.reward.numpy()).reshape(-1)[0])
            ep_reward  += reward
            ep_len     += 1
            total_steps += 1

            if ts.is_last():
                break

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)

    flip_rate = float(flip_count / max(total_steps, 1)) if attack else 0.0

    if attack and flip_rate == 0.0:
        print(
            f"  [WARN] Perturbator flip_rate=0 over {total_steps} steps. "
            "The perturbator did NOT change any action. "
            "Check that the correct perturbator.h5 checkpoint was loaded."
        )

    if log_actions and action_ids:
        uniq, counts = np.unique(np.asarray(action_ids, dtype=np.int32), return_counts=True)
        combo_stats  = []
        for action_id, count in zip(uniq, counts):
            prb_idx, sched_combo = cfg.actions[action_id]
            prb_alloc = cfg.feasible_prb_allocation_all[prb_idx]
            combo_stats.append((count, tuple(prb_alloc), tuple(sched_combo)))
        combo_stats.sort(reverse=True)
        print("Eval action combos (count):")
        for count, prb_alloc, sched_combo in combo_stats:
            print(f"  PRB [{','.join(map(str, prb_alloc))}] "
                  f"sched [{','.join(map(str, sched_combo))}]: {count}")

    return {
        "mean_reward":      float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "std_reward":       float(np.std(episode_rewards))  if episode_rewards else 0.0,
        "mean_ep_len":      float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "action_flip_rate": flip_rate,
    }


def run_eval_pair(
    perturbator: tf.keras.Model,
    actor_net,
    env,
    episodes: int,
    max_steps: int,
    label: str,
    action_mode: str = "sample",
) -> Dict[str, Dict]:
    clean  = evaluate_actor(
        perturbator, actor_net, env, episodes, max_steps,
        attack=False, action_mode=action_mode, log_actions=True,
    )
    attack = evaluate_actor(
        perturbator, actor_net, env, episodes, max_steps,
        attack=True, action_mode=action_mode, log_actions=True,
    )
    drop = clean["mean_reward"] - attack["mean_reward"]
    print(
        f"  [{label:5s}]  "
        f"mode={action_mode:6s}  "
        f"clean={clean['mean_reward']:8.4f}  "
        f"attacked={attack['mean_reward']:8.4f}  "
        f"drop={drop:8.4f}  "
        f"flip_rate={attack['action_flip_rate']:.4f}"
    )
    return {"clean": clean, "attack": attack}


def print_summary(pre: Dict, post: Dict) -> None:
    print("\n" + "=" * 60)
    print("  RADIAL-PPO ROBUSTNESS SUMMARY  (pre  vs  post-training)")
    print("=" * 60)
    print(f"  Clean reward   :  {pre['clean']['mean_reward']:8.4f}  →  {post['clean']['mean_reward']:8.4f}")
    print(f"  Attacked reward:  {pre['attack']['mean_reward']:8.4f}  →  {post['attack']['mean_reward']:8.4f}")
    pre_drop  = pre['clean']['mean_reward']  - pre['attack']['mean_reward']
    post_drop = post['clean']['mean_reward'] - post['attack']['mean_reward']
    print(f"  Reward drop    :  {pre_drop:8.4f}  →  {post_drop:8.4f}")
    improved  = post_drop < pre_drop
    preserved = abs(post['clean']['mean_reward'] - pre['clean']['mean_reward']) < abs(pre_drop) * 0.5
    print(f"  Robustness improved : {'YES ✓' if improved  else 'NO  ✗'}")
    print(f"  Clean perf preserved: {'YES ✓' if preserved else 'CHECK !'}")
    print("=" * 60 + "\n")


def maybe_load_actor_snapshot(actor_net, policy_dir: Optional[str]) -> Optional[dict]:
    paths     = load_snapshot_paths(policy_dir=policy_dir)
    meta      = read_json_if_exists(paths["meta"])
    actor_map = find_variable_map(meta, target="actor")
    print(f"Loading actor snapshot from: {paths['actor']}")
    load_npz_to_vars(paths["actor"], actor_net.variables, actor_map)
    if meta is not None:
        print("Loaded metadata:", meta)
    return meta


def save_training_outputs(out_dir, actor_net, logs, best_logs, args) -> None:
    os.makedirs(out_dir, exist_ok=True)

    actor_out      = os.path.join(out_dir, "actor.npz")
    best_actor_out = os.path.join(out_dir, "actor_best.npz")
    logs_out       = os.path.join(out_dir, "train_logs.npz")
    meta_out       = os.path.join(out_dir, "metadata.json")

    actor_variable_map = save_vars_to_npz(actor_out, actor_net.variables, prefix="actor")

    np.savez(
        logs_out,
        loss               = np.asarray(logs["loss"],               dtype=np.float32),
        ppo_loss           = np.asarray(logs["ppo_loss"],           dtype=np.float32),
        kl_norm            = np.asarray(logs["kl_norm"],            dtype=np.float32),
        kl_raw             = np.asarray(logs["kl_raw"],             dtype=np.float32),
        kl_clipped         = np.asarray(logs["kl_clipped"],         dtype=np.float32),
        entropy            = np.asarray(logs["entropy"],            dtype=np.float32),
        grad_norm          = np.asarray(logs["grad_norm"],          dtype=np.float32),
        current_eps        = np.asarray(logs["current_eps"],        dtype=np.float32),
        radial_weight_used = np.asarray(logs["radial_weight_used"], dtype=np.float32),
        mean_reward        = np.asarray(logs["mean_reward"],        dtype=np.float32),
        eval_clean_reward  = np.asarray(logs["eval_clean_reward"],  dtype=np.float32),
        eval_attack_reward = np.asarray(logs["eval_attack_reward"], dtype=np.float32),
    )

    metadata = {
        "mode":                    "radial_ppo",
        "init_policy_dir":         resolve_policy_dir(args.policy_dir),
        "perturbator_path":        args.perturbator_path,
        "eval_action_mode":        args.eval_action_mode,
        "actor_weights":           actor_out,
        "best_actor_weights":      best_actor_out if os.path.exists(best_actor_out) else "",
        "actor_variable_map":      actor_variable_map,
        "train_logs":              logs_out,
        "clean_weight":            args.clean_weight,
        "radial_weight":           args.radial_weight,
        "eps_start":               args.eps_start,
        "eps_final":               args.eps_final,
        "eps_schedule_start":      args.eps_schedule_start,
        "eps_schedule_end":        args.eps_schedule_end,
        "lr":                      args.lr,
        "max_kl_clip":             args.max_kl_clip,
        "no_adaptive_weight":      args.no_adaptive_weight,
        "adapt_window":            args.adapt_window,
        "adapt_drop_tol":          args.adapt_drop_tol,
        "best_eval_attack_reward": best_logs.get("best_eval_attack_reward"),
        "best_iter":               best_logs.get("best_iter"),
    }

    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved final actor  → {actor_out}")
    if os.path.exists(best_actor_out):
        print(f"Saved best actor   → {best_actor_out}")
    print(f"Saved logs         → {logs_out}")
    print(f"Saved metadata     → {meta_out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="RADIAL-PPO with learned perturbator")

    parser.add_argument(
        "--perturbator_path", default="pert.h5",
        help="Path to the learned perturbator Keras model (.h5). "
             "The model must accept [batch, obs_dim] float32 and return "
             "a tensor of the same shape."
    )

    parser.add_argument("--policy_dir", default="saved_policy/em-max/em-agent-filtered")
    parser.add_argument("--out_dir",        default="saved_policy/em-max/em-agent-radial")

    parser.add_argument("--train_iters",    type=int,   default=100)
    parser.add_argument("--rollout_horizon",type=int,   default=10)
    parser.add_argument("--gamma",          type=float, default=0.99)

    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--clip_ratio",     type=float, default=0.2)
    parser.add_argument("--entropy_coef",   type=float, default=0.01)
    parser.add_argument("--clean_weight",   type=float, default=0.5)
    parser.add_argument("--radial_weight",  type=float, default=0.1)
    parser.add_argument("--max_grad_norm",  type=float, default=0.9)

    parser.add_argument("--eps_start",          type=float, default=0.0)
    parser.add_argument("--eps_final",          type=float, default=0.3)
    parser.add_argument("--eps_schedule_start", type=int,   default=0)
    parser.add_argument("--eps_schedule_end",   type=int,   default=1000)

    parser.add_argument("--eval_every",     type=int, default=5)
    parser.add_argument("--eval_episodes",  type=int, default=5)
    parser.add_argument("--eval_max_steps", type=int, default=10)
    parser.add_argument("--eval_action_mode", choices=["sample", "greedy"], default="sample")

    parser.add_argument("--max_kl_clip",        type=float, default=1.0)
    parser.add_argument("--no_adaptive_weight", action="store_true")
    parser.add_argument("--adapt_window",       type=int,   default=20)
    parser.add_argument("--adapt_drop_tol",     type=float, default=0.10)
    parser.add_argument("--adapt_recover_rate", type=float, default=0.02)
    parser.add_argument("--adapt_floor",        type=float, default=0.01)

    parser.add_argument("--ibp_verify_tol", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=123) 
    args = parser.parse_args()

    set_all_seeds(args.seed)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ["ADVORAN_CONFIG_MODULE"] = "config_em_filtered"

    perturbator = load_perturbator(args.perturbator_path)

    env   = ran_env_wrapper.get_eval_env(config_obj=cfg)
    maybe_seed_env(env, args.seed)

    agent     = agent_builder.create_agent(env, algo="ppo", config_obj=cfg)
    actor_net = get_actor_net(agent)
    build_actor_once(env, actor_net)
    maybe_load_actor_snapshot(actor_net, args.policy_dir)

    print("\n========== IBP LAYER EXTRACTION ==========")
    dense_layers = extract_dense_layers(actor_net)
    if not dense_layers:
        raise RuntimeError(
            "No Dense layers found in actor_net.  "
            "RADIAL-PPO requires access to the network's Dense layers for IBP.  "
            "Check the layer extractor for your TF-Agents version."
        )
    print(f"Found {len(dense_layers)} Dense layer(s):")
    for i, lyr in enumerate(dense_layers):
        act_name = _activation_name(lyr.activation)
        print(f"  [{i}] {lyr.name}  kernel={tuple(lyr.kernel.shape)}  activation={act_name}")

    ts_verify  = env.reset()
    obs_verify = tf.convert_to_tensor(ts_verify.observation, dtype=tf.float32)
    st_verify  = ensure_step_type_shape(
        tf.convert_to_tensor(ts_verify.step_type, dtype=tf.int32)
    )
    verify_ibp_accuracy(actor_net, dense_layers, obs_verify, st_verify,
                        tol=args.ibp_verify_tol)
    print("==========================================\n")

    print("========== PRE-TRAINING EVALUATION ==========")
    pre = run_eval_pair(
        perturbator = perturbator,
        actor_net   = actor_net,
        env         = env,
        episodes    = args.eval_episodes,
        max_steps   = args.eval_max_steps,
        label       = "PRE",
        action_mode = args.eval_action_mode,
    )
    print("==============================================\n")

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, epsilon=1e-5)

    logs = {
        "loss": [], "ppo_loss": [], "kl_norm": [], "kl_raw": [], "kl_clipped": [],
        "entropy": [], "grad_norm": [], "current_eps": [], "radial_weight_used": [],
        "mean_reward": [], "eval_clean_reward": [], "eval_attack_reward": [],
    }
    adapt_state: Optional[Dict] = None
    best_logs = {"best_eval_attack_reward": -np.inf, "best_iter": -1}

    best_actor_path = os.path.join(args.out_dir, "actor_best.npz")
    os.makedirs(args.out_dir, exist_ok=True)

    for it in range(args.train_iters):
        eps_curr = curriculum_eps(
            iteration      = it,
            eps_final      = args.eps_final,
            eps_start      = args.eps_start,
            schedule_start = args.eps_schedule_start,
            schedule_end   = args.eps_schedule_end,
        )

        batch = collect_rollout(
            env       = env,
            agent     = agent,
            actor_net = actor_net,
            horizon   = args.rollout_horizon,
            gamma     = args.gamma,
        )

        mean_reward = float(np.mean(batch["rewards"]))
        if not args.no_adaptive_weight and len(logs["mean_reward"]) > 0:
            radial_w_curr, adapt_state = adaptive_radial_weight(
                base_weight    = args.radial_weight,
                reward_history = logs["mean_reward"],
                window         = args.adapt_window,
                drop_tol       = args.adapt_drop_tol,
                recover_rate   = args.adapt_recover_rate,
                floor          = args.adapt_floor,
                state          = adapt_state,
            )
        else:
            radial_w_curr = args.radial_weight

        stats = train_actor_step_radial(
            actor_net     = actor_net,
            dense_layers  = dense_layers,
            optimizer     = optimizer,
            obs           = tf.convert_to_tensor(batch["obs"],          dtype=tf.float32),
            step_type     = tf.convert_to_tensor(batch["step_type"],    dtype=tf.int32),
            actions       = tf.convert_to_tensor(batch["actions"],      dtype=tf.int32),
            advantages    = tf.convert_to_tensor(batch["advantages"],   dtype=tf.float32),
            old_log_probs = tf.convert_to_tensor(batch["old_log_probs"],dtype=tf.float32),
            clip_ratio    = args.clip_ratio,
            entropy_coef  = args.entropy_coef,
            clean_weight  = args.clean_weight,
            radial_weight = radial_w_curr,
            current_eps   = eps_curr,
            max_grad_norm = args.max_grad_norm,
            max_kl_clip   = args.max_kl_clip,
        )

        logs["loss"].append(stats["loss"])
        logs["ppo_loss"].append(stats["ppo_loss"])
        logs["kl_norm"].append(stats["kl_norm"])
        logs["kl_raw"].append(stats["kl_raw"])
        logs["kl_clipped"].append(stats["kl_clipped"])
        logs["entropy"].append(stats["entropy"])
        logs["grad_norm"].append(stats["grad_norm"])
        logs["current_eps"].append(eps_curr)
        logs["radial_weight_used"].append(radial_w_curr)
        logs["mean_reward"].append(mean_reward)

        print(
            f"[iter {it:04d}]  "
            f"eps={eps_curr:.4f}  rw={radial_w_curr:.4f}  "
            f"loss={stats['loss']:.4f}  "
            f"ppo={stats['ppo_loss']:.4f}  "
            f"kl_norm={stats['kl_norm']:.4f}  kl_raw={stats['kl_raw']:.2f}  "
            f"ent={stats['entropy']:.4f}  "
            f"gnorm={stats['grad_norm']:.4f}  "
            f"rew={mean_reward:.4f}"
        )

        do_eval = args.eval_every > 0 and (
            (it + 1) % args.eval_every == 0 or it == args.train_iters - 1
        )
        if do_eval:
            mid = run_eval_pair(
                perturbator = perturbator,
                actor_net   = actor_net,
                env         = env,
                episodes    = args.eval_episodes,
                max_steps   = args.eval_max_steps,
                label       = f"i{it:04d}",
                action_mode = args.eval_action_mode,
            )
            logs["eval_clean_reward"].append(mid["clean"]["mean_reward"])
            logs["eval_attack_reward"].append(mid["attack"]["mean_reward"])

            if mid["attack"]["mean_reward"] > best_logs["best_eval_attack_reward"]:
                best_logs["best_eval_attack_reward"] = mid["attack"]["mean_reward"]
                best_logs["best_iter"] = it
                save_vars_to_npz(best_actor_path, actor_net.variables, prefix="actor")
                print(f"           ↑ new best attacked reward; saved {best_actor_path}")
        else:
            logs["eval_clean_reward"].append(np.nan)
            logs["eval_attack_reward"].append(np.nan)

    save_training_outputs(
        out_dir       = args.out_dir,
        actor_net     = actor_net,
        logs          = logs,
        best_logs     = best_logs,
        args          = args,
    )

    print("\n========== POST-TRAINING EVALUATION ==========")
    post = run_eval_pair(
        perturbator = perturbator,
        actor_net   = actor_net,
        env         = env,
        episodes    = args.eval_episodes,
        max_steps   = args.eval_max_steps,
        label       = "POST",
        action_mode = args.eval_action_mode,
    )
    print("===============================================")

    print_summary(pre, post)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
