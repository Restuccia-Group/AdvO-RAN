from __future__ import annotations
from typing import List, Tuple
import numpy as np
import tensorflow as tf

from .structures import Trajectory
from .sla import SLAEvaluator, compute_vsla


class RewardModel(tf.keras.Model):


    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

        self.d1 = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.d2 = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.out = tf.keras.layers.Dense(1, activation=None)

    def call(self, s, a):

        if a.shape.ndims == 1:
            a_onehot = tf.one_hot(a, self.num_actions, dtype=tf.float32)
        else:
            a_onehot = tf.cast(a, tf.float32)
        x = tf.concat([s, a_onehot], axis=-1)
        x = self.d1(x)
        x = self.d2(x)
        r = self.out(x)         
        return tf.squeeze(r, axis=-1)  


def build_preference_pairs(
    trajectories: List[Trajectory],
    sla: SLAEvaluator,
    slice_type: str = "embb",
    dynamic: bool = True,
    min_vsla_diff: float = 0.05,
) -> List[Tuple[Trajectory, Trajectory, int]]:

    n = len(trajectories)
    vsla_values = [compute_vsla(tr, sla, slice_type, dynamic) for tr in trajectories]

    pairs: List[Tuple[Trajectory, Trajectory, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            x, z = vsla_values[i], vsla_values[j]
            diff = x - z
            if abs(diff) < min_vsla_diff:
                continue
            label = 1 if diff > 0 else 0
            pairs.append((trajectories[i], trajectories[j], label))
    return pairs


def make_preference_dataset(
    pairs: List[Tuple[Trajectory, Trajectory, int]],
    state_dim: int,
    seq_len: int,
    batch_size: int,
) -> tf.data.Dataset:


    def gen():
        for sigma_x, sigma_z, y in pairs:
            def traj_slice(traj: Trajectory):
                steps = traj.steps
                if len(steps) < seq_len:
                    steps = steps + [steps[-1]] * (seq_len - len(steps))
                else:
                    steps = steps[:seq_len]
                s = np.stack([st.s for st in steps], axis=0).astype(np.float32)
                a = np.array([st.a for st in steps], dtype=np.int32)
                return s, a

            s_x, a_x = traj_slice(sigma_x)
            s_z, a_z = traj_slice(sigma_z)

            yield (s_x, a_x, s_z, a_z, np.array(y, dtype=np.float32))

    spec = (
        tf.TensorSpec(shape=(seq_len, state_dim), dtype=tf.float32),  # s_x
        tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),              # a_x
        tf.TensorSpec(shape=(seq_len, state_dim), dtype=tf.float32),  # s_z
        tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),              # a_z
        tf.TensorSpec(shape=(), dtype=tf.float32),                    # y
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=spec)
    ds = ds.shuffle(buffer_size=len(pairs)).batch(batch_size)
    return ds



def train_reward_model_with_preferences(
    pairs: List[Tuple[Trajectory, Trajectory, int]],
    state_dim: int,
    num_actions: int,
    seq_len: int = 10,         
    batch_size: int = 16,
    hidden_dim: int = 128,
    num_epochs: int = 30,
    lr: float = 1e-3,
) -> RewardModel:

    ds = make_preference_dataset(
        pairs, state_dim=state_dim, seq_len=seq_len, batch_size=batch_size
    )

    reward_model = RewardModel(state_dim, num_actions, hidden_dim)
    optimizer = tf.keras.optimizers.Adam(lr)

    for epoch in range(num_epochs):
        for s_x, a_x, s_z, a_z, y in ds:
            B = tf.shape(s_x)[0]
            T = tf.shape(s_x)[1]

            s_x_flat = tf.reshape(s_x, [B * T, state_dim])
            a_x_flat = tf.reshape(a_x, [B * T])
            s_z_flat = tf.reshape(s_z, [B * T, state_dim])
            a_z_flat = tf.reshape(a_z, [B * T])

            with tf.GradientTape() as tape:
                r_x_flat = reward_model(s_x_flat, a_x_flat)  # [B*T]
                r_z_flat = reward_model(s_z_flat, a_z_flat)

                R_x = tf.reshape(r_x_flat, [B, T])
                R_z = tf.reshape(r_z_flat, [B, T])

                score_x = tf.reduce_sum(R_x, axis=1)  # [B]
                score_z = tf.reduce_sum(R_z, axis=1)

                logits = score_x - score_z
                p_x_pref = tf.nn.sigmoid(logits)

                loss = tf.keras.losses.binary_crossentropy(y, p_x_pref)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, reward_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, reward_model.trainable_variables))

        print(f"[PbRL] Epoch {epoch+1}/{num_epochs} | loss = {loss.numpy():.4f}")

    return reward_model