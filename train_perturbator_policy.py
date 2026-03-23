

import argparse
import glob
import json
import os
 
import numpy as np
import tensorflow as tf
 
import agent_builder
import ran_env_wrapper
import config_em_filtered as cfg
 
 
DEFAULT_POLICY_DIR_CANDIDATES = [
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


def read_json_if_exists(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Warning: could not read metadata {path}: {exc}")
        return None


def find_variable_map(meta, target="actor"):
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
    for value in meta.values():
        if isinstance(value, dict):
            for key in candidate_keys:
                if key in value and isinstance(value[key], list):
                    return value[key]
    return None


def load_npz_to_vars(npz_path, variables, variable_map=None):
    data = np.load(npz_path)
    if variable_map:
        name_to_var = {v.name: v for v in variables}
        loaded = 0
        for item in variable_map:
            saved_key = item.get("saved_key")
            var_name = item.get("var_name")
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
        print("Metadata map matched no variables; falling back to raw order.")

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
        _resolve_snapshot_file(policy_dir, "metadata", "json", required=False),
    )
 
 
def load_actor(eval_env, policy_dir, label="actor"):
    agent = agent_builder.create_agent(eval_env, algo="ppo", config_obj=cfg)
    actor_path, value_path, _, meta_path = load_snapshot(policy_dir)
    meta = read_json_if_exists(meta_path)
    actor_net = getattr(agent, "actor_net", getattr(agent, "_actor_net"))
    value_net = getattr(agent, "value_net", getattr(agent, "_value_net"))
    load_npz_to_vars(actor_path, actor_net.variables, find_variable_map(meta, target="actor"))
    load_npz_to_vars(value_path, value_net.variables, find_variable_map(meta, target="value"))
    print(f"Loaded {label} from {os.path.dirname(actor_path)}")
    if meta is not None:
        print(f"  metadata ({meta_path}): {meta}")
    return agent, actor_net
 
 
def decode_action(action_id):
    from config_em_filtered import actions, feasible_prb_allocation_all, scheduling_combos
    a = actions[action_id]
    return feasible_prb_allocation_all[a[0]], a[1]
 
 
def get_probs(dist):
    """Safely extract probability vector from a Categorical distribution."""
    try:
        return dist.probs_parameter()
    except Exception:
        return tf.nn.softmax(dist.logits_parameter(), axis=-1)
 
 
def load_reward_model(path: str) -> tf.keras.Model:

    model = tf.keras.models.load_model(path, compile=False)
    model.trainable = False
    print(f"Loaded reward model : {path}")
    print(f"  input  shape : {model.input_shape}")
    print(f"  output shape : {model.output_shape}")
    return model
 
 
def build_action_reward_inputs() -> tf.Tensor:

    from config_em_filtered import actions
 
    num_actions = len(actions)
    raw_prb   = np.zeros(num_actions, dtype=np.float32)
    raw_sched = np.zeros(num_actions, dtype=np.float32)
 
    for a_id in range(num_actions):
        prb_alloc, sched = decode_action(a_id)
 
        if a_id < 6:                        # debug: inspect first few actions
            print(f"  [debug] action {a_id}: prb_alloc={prb_alloc!r}  sched={sched!r}")
 
        prb_arr   = np.asarray(prb_alloc, dtype=np.float32).flatten()
        sched_arr = np.asarray(sched,     dtype=np.float32).flatten()
        raw_prb[a_id]   = float(prb_arr[0])   if prb_arr.size   > 0 else 0.0
        raw_sched[a_id] = float(sched_arr[0]) if sched_arr.size > 0 else 0.0
 
    print(f"  [debug] raw_prb   unique : {np.unique(raw_prb)}")
    print(f"  [debug] raw_sched unique : {np.unique(raw_sched)}")
 
    def minmax(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-8)
 
    norm_prb   = minmax(raw_prb)
    norm_sched = minmax(raw_sched)
 
    action_inputs = np.stack([norm_prb, norm_sched], axis=1)      # (A, 2)

    return tf.constant(action_inputs, dtype=tf.float32)            # frozen
 
 
def build_perturbator_model(obs_dim, eps, hidden, obs_mean, obs_std):
    mean_list = np.asarray(obs_mean, dtype=np.float32).tolist()
    std_list = np.asarray(obs_std, dtype=np.float32).tolist()
    inp = tf.keras.Input(shape=(obs_dim,), dtype=tf.float32, name="obs_input")
    norm = tf.keras.layers.Lambda(
        lambda x, mean=mean_list, std=std_list: (x - tf.constant(mean)) / (tf.constant(std) + 1e-8),
        name="normalize",
    )(inp)
    x = norm
    for idx, width in enumerate(hidden):
        x = tf.keras.layers.Dense(width, activation="relu", name=f"hidden_{idx}")(x)
    delta = tf.keras.layers.Dense(obs_dim, name="delta_out")(x)
    clipped = tf.keras.layers.Lambda(
        lambda y: tf.clip_by_value(y, -eps, eps),
        name="clip",
    )(delta)
    return tf.keras.Model(inputs=inp, outputs=clipped, name="perturbator")


def adversary_loss(
    delta,
    obs_b,
    victim_actor,
    ref_actor,
    step_type,
    init_state_v,
    init_state_r,
    reward_model,
    action_reward_inputs,
    eps,
    lam,
    beta,
    gamma_w,
    p_norm,
):

    adv_obs = obs_b + delta
 
    dist_v, _ = victim_actor(adv_obs, step_type, init_state_v, training=False)
    dist_r, _ = ref_actor(obs_b,      step_type, init_state_r, training=False)
 
    victim_probs    = get_probs(dist_v)                                # (B, A)
    ref_probs       = get_probs(dist_r)                                # (B, A)
    victim_log_prob = tf.math.log(victim_probs + 1e-8)                 # (B, A)
    ref_log_prob    = tf.math.log(ref_probs    + 1e-8)                 # (B, A)
 
    if p_norm == np.inf:
        norms = tf.reduce_max(tf.abs(delta), axis=-1)
    else:
        norms = tf.norm(delta, ord=p_norm, axis=-1)
    l_pen = lam * tf.reduce_mean(tf.nn.relu(norms - eps))
 
    kl_per_sample = tf.reduce_sum(
        victim_probs * (victim_log_prob - ref_log_prob), axis=-1       # (B,)
    )
    l_kl = beta * tf.reduce_mean(kl_per_sample)
 

    if gamma_w > 0:
        r_hat = reward_model(action_reward_inputs, training=False)     # (A, 1)
        r_hat = tf.stop_gradient(tf.reshape(r_hat, [-1]))             # (A,)
 
        j_per_sample = tf.reduce_sum(
            victim_probs * r_hat[tf.newaxis, :], axis=-1              # (B,)
        )
        j_loss = -gamma_w * tf.reduce_mean(j_per_sample)
    else:
        j_loss = tf.constant(0.0)
 
    loss = j_loss + l_pen + l_kl
    return loss, j_loss, l_pen, l_kl
 
 
def collect_observations(eval_env, agent, num_steps):
    """Roll out the victim policy; return observations and immediate rewards."""
    obs_list, rew_list = [], []
    time_step = eval_env.reset()
    for _ in range(num_steps):
        obs_list.append(time_step.observation)
        rew_list.append(
            np.full((time_step.observation.shape[0],), float(time_step.reward))
        )
        action_step = agent.policy.action(time_step)
        time_step   = eval_env.step(action_step.action)
        if time_step.is_last():
            time_step = eval_env.reset()
    return (
        np.concatenate(obs_list, axis=0),
        np.concatenate(rew_list, axis=0),
    )
 
 
def train(
    perturb_net, victim_actor, ref_actor,
    observations,
    reward_model, action_reward_inputs,
    epochs=100, batch_size=64, lr=1e-3,
    lam=0.1, beta=1.0, gamma_w=0.5, p_norm=2,
):

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    dataset = (
        tf.data.Dataset.from_tensor_slices(observations.astype(np.float32))
        .shuffle(len(observations))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    history = []
    for epoch in range(1, epochs + 1):
        epoch_losses = {"total": [], "J": [], "L_pen": [], "L_KL": []}
        for obs_b in dataset:
            b            = tf.shape(obs_b)[0]
            step_type    = tf.ones((b,), dtype=tf.int32)
            init_state_v = victim_actor.get_initial_state(batch_size=b)
            init_state_r = ref_actor.get_initial_state(batch_size=b)
 
            with tf.GradientTape() as tape:
                delta = perturb_net(obs_b, training=True)
                loss, j_loss, l_pen, l_kl = adversary_loss(
                    delta                = delta,
                    obs_b                = obs_b,
                    victim_actor         = victim_actor,
                    ref_actor            = ref_actor,
                    step_type            = step_type,
                    init_state_v         = init_state_v,
                    init_state_r         = init_state_r,
                    reward_model         = reward_model,
                    action_reward_inputs = action_reward_inputs,
                    eps                  = perturb_net.eps,
                    lam                  = lam,
                    beta                 = beta,
                    gamma_w              = gamma_w,
                    p_norm               = p_norm,
                )
            grads = tape.gradient(loss, perturb_net.trainable_variables)
            optimizer.apply_gradients(zip(grads, perturb_net.trainable_variables))
 
            epoch_losses["total"].append(float(loss))
            epoch_losses["J"].append(float(j_loss))
            epoch_losses["L_pen"].append(float(l_pen))
            epoch_losses["L_KL"].append(float(l_kl))
 
        means = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
        history.append(means)
        if epoch % 20 == 0 or epoch == 1:
            print(
                f"  [epoch {epoch:>4}/{epochs}]  "
                f"total={means['total']:.4f}  "
                f"J={means['J']:.4f}  "
                f"L_pen={means['L_pen']:.4f}  "
                f"L_KL={means['L_KL']:.4f}"
            )
    return history
 
 
def run_step(time_step, obs, perturb_net, victim_agent, ref_actor,
             is_attack, step_idx, phase):
    if is_attack and perturb_net is not None:
        delta   = perturb_net(obs, training=False)
        adv_obs = obs + delta
        linf    = float(np.max(np.abs(delta.numpy())))
    else:
        adv_obs = obs
        linf    = 0.0
 
    attacked_ts = time_step._replace(observation=adv_obs)
    action_step = victim_agent.policy.action(attacked_ts)
    victim_id   = int(action_step.action.numpy().flatten()[0])
 
    b            = tf.shape(obs)[0]
    step_type    = tf.convert_to_tensor(time_step.step_type, dtype=tf.int32)
    init_state_r = ref_actor.get_initial_state(batch_size=b)
    dist_r, _    = ref_actor(obs, step_type, init_state_r, training=False)
    ref_id       = int(tf.argmax(get_probs(dist_r), axis=-1).numpy().flatten()[0])
 
    reward = float(time_step.reward)
    match  = "match" if victim_id == ref_id else "miss"
    prb_alloc, sched = decode_action(victim_id)
    print(
        f"[{phase} t={step_idx}] victim={victim_id} ref={ref_id} [{match}] "
        f"prb={prb_alloc} sched={sched} reward={reward:.4f} L_inf={linf:.4f}"
    )
    return action_step, victim_id, ref_id, reward
 
 
def evaluate(eval_env, victim_agent, victim_actor, ref_actor, perturb_net, horizon):
    def run_phase(phase, is_attack):
        time_step = eval_env.reset()
        rewards, matches, total = [], 0, 0
        for t in range(horizon):
            obs = tf.convert_to_tensor(time_step.observation, dtype=tf.float32)
            action_step, victim_id, ref_id, reward = run_step(
                time_step, obs, perturb_net, victim_agent, ref_actor,
                is_attack=is_attack, step_idx=t, phase=phase,
            )
            rewards.append(reward)
            matches += int(victim_id == ref_id)
            total   += 1
            time_step = eval_env.step(action_step.action)
            if time_step.is_last():
                time_step = eval_env.reset()
        return sum(rewards), matches, total
 
    print("\n" + "=" * 64)
    print("Phase 1 — Baseline (clean victim, fresh env)")
    print("=" * 64)
    base_r, base_m, base_t = run_phase("baseline", is_attack=False)
 
    print("\n" + "=" * 64)
    print("Phase 2 — Attack (perturbed victim, fresh env)")
    print("=" * 64)
    atk_r, atk_m, atk_t = run_phase("attack", is_attack=True)
 
    print("\n" + "=" * 64)
    print("Phase 3 — Recovery (clean victim, fresh env)")
    print("=" * 64)
    rec_r, rec_m, rec_t = run_phase("recovery", is_attack=False)
 
    safe = base_r if abs(base_r) > 1e-8 else 1e-8
    print("\n" + "=" * 64)
    print("Summary")
    print("=" * 64)
    print(f"  Baseline  reward : {base_r:>10.4f}  | ref-match {base_m}/{base_t} ({100*base_m/base_t:.1f}%)")
    print(f"  Attack    reward : {atk_r:>10.4f}  | ref-match {atk_m}/{atk_t} ({100*atk_m/atk_t:.1f}%)  "
          f"| reward delta {100*(base_r-atk_r)/abs(safe):+.1f}%")
    print(f"  Recovery  reward : {rec_r:>10.4f}  | ref-match {rec_m}/{rec_t} ({100*rec_m/rec_t:.1f}%)")
    print("=" * 64)
    print(f"  Attack effectiveness (ref-match lift): "
          f"{atk_m/atk_t:.3f} - {base_m/base_t:.3f} = {(atk_m/atk_t)-(base_m/base_t):+.3f}")
    print("  Recovery ref-match ~= Baseline ref-match => victim undamaged.")
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Amortized adversarial attack — direct eq 11 loss (AdvO-RAN)"
    )
    parser.add_argument("--policy_dir",       default=None)
    parser.add_argument("--ref_policy_dir",   default=None)
    parser.add_argument("--reward_model",     default="reward_model.h5",
                        help="Path to reward_model.h5. "
                             "Input: normalised [prb, sched] shape (B, 2). "
                             "Output: scalar reward estimate shape (B, 1).")
    parser.add_argument("--eps",              type=float, default=0.3)
    parser.add_argument("--p_norm",           type=float, default=float("inf"),
                        help="p for L_p norm in L_pen (eq 9). 2=L2, inf=L_inf.")
    parser.add_argument("--lam",              type=float, default=1.0,
                        help="lambda: L_pen weight (eq 9)")
    parser.add_argument("--beta",             type=float, default=0.2,
                        help="beta: L_KL weight (eq 10)")
    parser.add_argument("--gamma_w",          type=float, default=1.0,
                        help="gamma_w: J weight (eq 8). 0 = disable J.")
    parser.add_argument("--collect_steps",    type=int,   default=1000)
    parser.add_argument("--epochs",           type=int,   default=100)
    parser.add_argument("--batch_size",       type=int,   default=64)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--hidden",           type=int,   nargs="+", default=[256, 256])
    parser.add_argument("--horizon",          type=int,   default=10)
    parser.add_argument("--no_attack",        action="store_true")
    parser.add_argument("--save_perturb_net", default="perturbator.keras",
                        help="Path to save the perturbation network (Keras format)")
    args = parser.parse_args()
 
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ["ADVORAN_CONFIG_MODULE"] = "config_em_filtered"
 
    p_norm = np.inf if args.p_norm == float("inf") else args.p_norm
 
    eval_env   = ran_env_wrapper.get_eval_env(config_obj=cfg)
    time_step  = eval_env.reset()
    obs_sample = tf.convert_to_tensor(time_step.observation, dtype=tf.float32)
    obs_dim    = int(obs_sample.shape[-1])
 
    victim_agent, victim_actor = load_actor(
        eval_env, args.policy_dir, label="victim"
    )
    _, ref_actor = load_actor(
        eval_env, args.ref_policy_dir or args.policy_dir, label="reference"
    )
 
    reward_model         = load_reward_model(args.reward_model)
    action_reward_inputs = build_action_reward_inputs()    # (A, 2), constant
 
    print(f"\nobs_dim={obs_dim}  eps={args.eps}  "
          f"lam={args.lam}  beta={args.beta}  gamma_w={args.gamma_w}  p={p_norm}")
    print(
        f"victim_dir={resolve_policy_dir(args.policy_dir)}  "
        f"reference_dir={resolve_policy_dir(args.ref_policy_dir or args.policy_dir)}"
    )
    print(f"Loss: L = J(eq8, r_hat) + L_pen(eq9) + L_KL(eq10)  [direct, no PGD]\n")
 
    if args.no_attack:
        evaluate(eval_env, victim_agent, victim_actor, ref_actor,
                 perturb_net=None, horizon=args.horizon)
        return
 
    print(f"Collecting {args.collect_steps} observations ...")
    observations, rewards = collect_observations(
        eval_env, victim_agent, args.collect_steps
    )
    print(f"Dataset: obs={observations.shape}  "
          f"rewards mean={rewards.mean():.4f}  std={rewards.std():.4f}")
 
    obs_mean = observations.mean(axis=0)
    obs_std  = observations.std(axis=0)
 
    perturb_net = build_perturbator_model(
        obs_dim=obs_dim,
        eps=args.eps,
        hidden=tuple(args.hidden),
        obs_mean=obs_mean,
        obs_std=obs_std,
    )
    perturb_net.eps = float(args.eps)
    _ = perturb_net(obs_sample)     # build weights
    print(f"PerturbationNet params: "
          f"{sum(int(np.prod(v.shape)) for v in perturb_net.trainable_variables):,}")
 
    print(f"\nTraining PerturbationNet with direct combined loss (eqs 8-11) ...")
    train(
        perturb_net          = perturb_net,
        victim_actor         = victim_actor,
        ref_actor            = ref_actor,
        observations         = observations,
        reward_model         = reward_model,
        action_reward_inputs = action_reward_inputs,
        epochs               = args.epochs,
        batch_size           = args.batch_size,
        lr                   = args.lr,
        lam                  = args.lam,
        beta                 = args.beta,
        gamma_w              = args.gamma_w,
        p_norm               = p_norm,
    )
 
    if args.save_perturb_net:
        save_path = args.save_perturb_net
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        perturb_net.save(save_path, include_optimizer=False)
        print(f"Saved model -> {save_path}")
 
    evaluate(
        eval_env     = eval_env,
        victim_agent = victim_agent,
        victim_actor = victim_actor,
        ref_actor    = ref_actor,
        perturb_net  = perturb_net,
        horizon      = args.horizon,
    )
 
 
if __name__ == "__main__":
    raise SystemExit(main())
