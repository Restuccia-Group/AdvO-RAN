

import argparse
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# ─────────────────────────── SCHEDULING MAPS ──────────────────────────────────
SCHED_INT_TO_NAME = {0: "RR", 1: "PF", 2: "WF"}
SCHED_STR_TO_INT  = {"RR": 0, "PF": 1, "WF": 2}
N_SCHED           = 3

# ─────────────────────────── HYPERPARAMETERS ──────────────────────────────────
MAX_KBPS        = 4000000.0   # 4 Mbps
SLA_THRESHOLD   = 0.70     # violation if dl_bitrate_norm < 0.70
ROWS_PER_STATE  = 10       # rows sampled per time-step
STATES_PER_TRAJ = 10       # d states per trajectory
N_PAIRS         = 2_000
BATCH_SIZE      = 64
LR              = 1e-3
EPOCHS          = 50
HIDDEN_DIM      = 64
VAL_SPLIT       = 0.20
SEED            = 42
STATE_DIM       = 2        # [prb_norm, sched_norm]

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

MAX_PRB = 50 #15  


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def normalise_sched(series):
    """Accept int {0,1,2} OR string {"RR","PF","WF"} -> int Series."""
    sample = series.dropna().iloc[0]
    if isinstance(sample, str):
        return series.str.strip().str.upper().map(SCHED_STR_TO_INT)
    return series.astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 │ LOAD & PRE-PROCESS
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(csv_path):
    global MAX_PRB

    df = pd.read_csv(csv_path)

    # Validate columns
    required = {"dl_brate", "slice_prb", "sched_alg"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}\nFound: {list(df.columns)}")

    # Scheduling -> int {0,1,2}
    df["sched_int"] = normalise_sched(df["sched_alg"])
    if df["sched_int"].isna().any():
        bad = df["sched_alg"][df["sched_int"].isna()].unique()
        raise ValueError(f"Unrecognised sched_alg values: {bad}")
    df["sched_name"] = df["sched_int"].map(SCHED_INT_TO_NAME)

    # Derive MAX_PRB from data
    MAX_PRB = float(df["slice_prb"].max())

    # Normalise bitrate (Kbps -> [0,1])
    df["dl_bitrate_norm"] = (df["dl_brate"].astype(float) / MAX_KBPS).clip(0.0, 1.0)

    # Static SLA violation label (used ONLY for labelling, NOT model input)
    df["sla_violation"] = (df["dl_bitrate_norm"] < SLA_THRESHOLD).astype(float)

    # Action key
    df["action_key"] = df["slice_prb"].astype(str) + "_" + df["sched_name"]

    # Diagnostics
    print(f"\n{'='*60}")
    print(f"  CSV        : {csv_path}")
    print(f"  Rows       : {len(df):,}")
    print(f"  MAX_KBPS   : {MAX_KBPS:.0f} Kbps  (= 4 Mbps)")
    print(f"  MAX_PRB    : {MAX_PRB:.0f}  (auto from data)")
    print(f"  SLA thresh : {SLA_THRESHOLD*MAX_KBPS:.0f} Kbps  ({SLA_THRESHOLD:.0%} of max)")
    print(f"  Model input: [prb_norm, sched_norm]  (STATE_DIM = {STATE_DIM})")
    print(f"\n  dl_brate stats:")
    print(f"    min  = {df['dl_brate'].min():.2f} Kbps")
    print(f"    mean = {df['dl_brate'].mean():.2f} Kbps")
    print(f"    max  = {df['dl_brate'].max():.2f} Kbps")
    print(f"\n  PRB values in data : {sorted(df['slice_prb'].unique())}")
    print(f"  Global SLA violation rate : {df['sla_violation'].mean():.2%}")
    print(f"\n  {'Action':<12}  {'Rows':>6}  {'SLA viol%':>10}")
    print(f"  {'─'*32}")
    for ak in sorted(df["action_key"].unique()):
        sub = df[df["action_key"] == ak]
        print(f"  {ak:<12}  {len(sub):>6}  {sub['sla_violation'].mean():>10.2%}")
    print(f"{'='*60}\n")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 │ ACTION FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_action_features(rows):
    """
    [prb_norm, sched_norm]  shape: (2,)
    All rows share the same action so just read from first row.
    """
    return np.array([
        float(rows["slice_prb"].iloc[0]) / MAX_PRB,
        float(rows["sched_int"].iloc[0]) / (N_SCHED - 1),
    ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 │ TRAJECTORY SAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def sample_trajectory(df):

    action  = random.choice(df["action_key"].unique())
    adf     = df[df["action_key"] == action].reset_index(drop=True)
    replace = len(adf) < ROWS_PER_STATE

    action_feat = extract_action_features(adf)   # (2,)

    flags = []
    for _ in range(STATES_PER_TRAJ):
        rows = adf.sample(ROWS_PER_STATE, replace=replace)
        flags.append(1.0 if rows["sla_violation"].mean() > 0 else 0.0)

    # Same action repeated T times -> (T, 2)
    states = np.tile(action_feat, (STATES_PER_TRAJ, 1)).astype(np.float32)

    return {
        "states"          : states,
        "step_viol_flags" : np.array(flags, dtype=np.float32),
        "action"          : action,
    }


def compute_v_sla(traj):
    """V_sla(sigma) = mean(I_sigma(t))   [Eq. 2]"""
    return float(traj["step_viol_flags"].mean())


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 │ PREFERENCE DATASET
# ══════════════════════════════════════════════════════════════════════════════

def build_preference_dataset(df, n_pairs=N_PAIRS):
    """
    Build D = {(sigma_x, sigma_z, y)}.

    y = [1,0]  sigma_x preferred (more SLA violations)
    y = [0,1]  sigma_z preferred
    tie        random
    """
    sx_list, sz_list, y_list = [], [], []
    n_ties = 0

    for i in range(n_pairs):
        tx = sample_trajectory(df)
        tz = sample_trajectory(df)
        vx = compute_v_sla(tx)
        vz = compute_v_sla(tz)

        if vx > vz:
            y = [1.0, 0.0]
        elif vz > vx:
            y = [0.0, 1.0]
        else:
            y = random.choice([[1.0, 0.0], [0.0, 1.0]])
            n_ties += 1

        sx_list.append(tx["states"])   # (T, 2)
        sz_list.append(tz["states"])   # (T, 2)
        y_list.append(y)               # (2,)

        if (i + 1) % 500 == 0:
            print(f"  [{i+1:>5}/{n_pairs}] pairs built  (ties: {n_ties})")

    print(f"  Done: {n_pairs} pairs  |  ties: {n_ties} ({n_ties/n_pairs:.1%})\n")

    return (
        np.array(sx_list, dtype=np.float32),   # (N, T, 2)
        np.array(sz_list, dtype=np.float32),   # (N, T, 2)
        np.array(y_list,  dtype=np.float32),   # (N, 2)
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 │ REWARD MODEL  (Keras)
# ══════════════════════════════════════════════════════════════════════════════

def build_reward_model(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM):
    """
    Simple MLP: [prb_norm, sched_norm] -> scalar reward

    Input shape : (state_dim,)  = (2,)
    Output      : (1,)  scalar
    """
    inp = keras.Input(shape=(state_dim,), name="action_input")
    x   = keras.layers.Dense(hidden_dim, name="dense_1")(inp)
    x   = keras.layers.LayerNormalization(name="ln_1")(x)
    x   = keras.layers.ReLU(name="relu_1")(x)
    x   = keras.layers.Dense(hidden_dim, name="dense_2")(x)
    x   = keras.layers.ReLU(name="relu_2")(x)
    x   = keras.layers.Dense(hidden_dim // 2, name="dense_3")(x)
    x   = keras.layers.ReLU(name="relu_3")(x)
    out = keras.layers.Dense(1, name="reward_out")(x)   # scalar r_hat

    model = keras.Model(inputs=inp, outputs=out, name="reward_model")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 │ BRADLEY-TERRY LOSS  (custom Keras loss)
# ══════════════════════════════════════════════════════════════════════════════

class BradleyTerryLoss(keras.losses.Loss):
    """
    L(psi) = -E[ sum_{i in {x,z}} y^(i) * log P_psi[sigma_i > sigma_{-i}] ]

    Inputs to call():
        y_true : (B, 2)   one-hot preference label
        y_pred : (B, 2)   [R_x, R_z]  cumulative trajectory rewards
    """
    def call(self, y_true, y_pred):
        log_probs = tf.nn.log_softmax(y_pred, axis=1)   # (B, 2)
        loss = -tf.reduce_sum(y_true * log_probs, axis=1)
        return tf.reduce_mean(loss)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 │ TRAINING  (custom loop - needed for paired trajectory input)
# ══════════════════════════════════════════════════════════════════════════════

def train_reward_model(sx_all, sz_all, y_all,
                       epochs=EPOCHS, batch_sz=BATCH_SIZE, lr=LR):
    """
    sx_all : (N, T, 2)
    sz_all : (N, T, 2)
    y_all  : (N, 2)

    Custom training loop because the loss needs both trajectories simultaneously.
    """
    # Train / val split
    n       = len(y_all)
    n_val   = max(1, int(n * VAL_SPLIT))
    idx     = np.random.permutation(n)
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    sx_tr, sz_tr, y_tr = sx_all[tr_idx], sz_all[tr_idx], y_all[tr_idx]
    sx_vl, sz_vl, y_vl = sx_all[val_idx], sz_all[val_idx], y_all[val_idx]

    model     = build_reward_model()
    optimizer = keras.optimizers.Adam(learning_rate=lr, weight_decay=1e-5)
    bt_loss   = BradleyTerryLoss()

    n_tr  = len(tr_idx)
    steps = int(np.ceil(n_tr / batch_sz))

    print(f"  Train pairs: {n_tr}  |  Val pairs: {n_val}\n")
    print(f"  {'Epoch':>5}  {'Train Loss':>12}  {'Val Loss':>10}  "
          f"{'Val Acc':>8}  Trend")
    print(f"  {'─'*55}")

    prev_val = None

    for epoch in range(1, epochs + 1):

        # ── Shuffle training data each epoch ──────────────────────────────────
        perm = np.random.permutation(n_tr)
        sx_tr, sz_tr, y_tr = sx_tr[perm], sz_tr[perm], y_tr[perm]

        # ── Training batches ──────────────────────────────────────────────────
        t_loss_sum = 0.0
        for step in range(steps):
            sl  = slice(step * batch_sz, (step + 1) * batch_sz)
            bsx = tf.constant(sx_tr[sl])   # (B, T, 2)
            bsz = tf.constant(sz_tr[sl])
            by  = tf.constant(y_tr[sl])    # (B, 2)

            with tf.GradientTape() as tape:
                # Cumulative reward per trajectory: sum over T steps
                # model input: (B*T, 2)  -> output (B*T, 1) -> reshape (B, T) -> sum -> (B,)
                B = tf.shape(bsx)[0]
                R_x = tf.reduce_sum(
                    tf.reshape(model(tf.reshape(bsx, (-1, STATE_DIM))), (B, STATES_PER_TRAJ)),
                    axis=1
                )   # (B,)
                R_z = tf.reduce_sum(
                    tf.reshape(model(tf.reshape(bsz, (-1, STATE_DIM))), (B, STATES_PER_TRAJ)),
                    axis=1
                )   # (B,)
                logits = tf.stack([R_x, R_z], axis=1)   # (B, 2)
                loss   = bt_loss(by, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            grads = [tf.clip_by_norm(g, 1.0) for g in grads]
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            t_loss_sum += loss.numpy()

        t_loss = t_loss_sum / steps

        # ── Validation ────────────────────────────────────────────────────────
        bsx = tf.constant(sx_vl)
        bsz = tf.constant(sz_vl)
        by  = tf.constant(y_vl)
        B   = tf.shape(bsx)[0]

        R_x = tf.reduce_sum(
            tf.reshape(model(tf.reshape(bsx, (-1, STATE_DIM)), training=False),
                       (B, STATES_PER_TRAJ)), axis=1
        )
        R_z = tf.reduce_sum(
            tf.reshape(model(tf.reshape(bsz, (-1, STATE_DIM)), training=False),
                       (B, STATES_PER_TRAJ)), axis=1
        )
        logits  = tf.stack([R_x, R_z], axis=1)
        v_loss  = bt_loss(by, logits).numpy()
        pred    = tf.cast(R_x > R_z, tf.float32)
        v_acc   = tf.reduce_mean(tf.cast(pred == y_vl[:, 0], tf.float32)).numpy()

        trend    = "  —" if prev_val is None else ("  ↓" if v_loss < prev_val else "  ↑")
        prev_val = v_loss

        print(f"  {epoch:>5}  {t_loss:>12.5f}  {v_loss:>10.5f}  "
              f"{v_acc:>7.2%}  {trend}")

    print(f"  {'─'*55}\n")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 │ PER-ACTION REWARD TABLE
# ══════════════════════════════════════════════════════════════════════════════

def scalar_reward(model, prb, sched_int):
    """r_hat for a single (prb, sched) action."""
    feat = np.array([[
        float(prb) / MAX_PRB,
        float(sched_int) / (N_SCHED - 1),
    ]], dtype=np.float32)   # (1, 2)
    return float(model(feat, training=False).numpy()[0, 0])


def evaluate_per_action_rewards(model, df):
    print("\n  Per-action reward  (higher r_hat = attacker prefers it):")
    print(f"  {'Action':<12}  {'Rows':>6}  {'SLA viol%':>10}  {'r_hat':>8}")
    print(f"  {'─'*44}")

    seen = set()
    for ak in sorted(df["action_key"].unique()):
        sub     = df[df["action_key"] == ak]
        prb     = int(sub["slice_prb"].iloc[0])
        sched_i = int(sub["sched_int"].iloc[0])
        if (prb, sched_i) in seen:
            continue
        seen.add((prb, sched_i))
        r = scalar_reward(model, prb, sched_i)
        print(f"  {ak:<12}  {len(sub):>6}  "
              f"{sub['sla_violation'].mean():>10.2%}  {r:>8.4f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="AdvO-RAN EM-agent Reward Model (Keras)")
    p.add_argument("--csv",       type=str,   required=True, help="Path to CSV")
    p.add_argument("--n_pairs",   type=int,   default=N_PAIRS)
    p.add_argument("--epochs",    type=int,   default=EPOCHS)
    p.add_argument("--lr",        type=float, default=LR)
    p.add_argument("--out_model", type=str,   default="reward_model.h5")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("\n[1/4]  Loading & pre-processing …")
    df = load_and_preprocess(args.csv)

    print("[2/4]  Building preference dataset …")
    sx, sz, y = build_preference_dataset(df, n_pairs=args.n_pairs)

    print("[3/4]  Training reward model …")
    model = train_reward_model(sx, sz, y, epochs=args.epochs, lr=args.lr)

    print("[4/4]  Saving & evaluating …")
    model.save(args.out_model)
    print(f"  Model saved -> {args.out_model}")

    evaluate_per_action_rewards(model, df)
    print("  Done.\n")