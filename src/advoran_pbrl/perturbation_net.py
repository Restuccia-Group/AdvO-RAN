import numpy as np
import tensorflow as tf
import gym
from typing import List

from tf_agents.trajectories import time_step as ts

from .structures import Trajectory
from .reward_model import RewardModel


class PerturbationNet(tf.keras.Model):


    def __init__(self, state_dim: int, hidden_dims=(128, 128), epsilon: float = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.epsilon = float(epsilon)

        self.hidden_layers = [
            tf.keras.layers.Dense(h, activation="relu") for h in hidden_dims
        ]
        self.out_layer = tf.keras.layers.Dense(state_dim, activation=None)

    def call(self, s: tf.Tensor) -> tf.Tensor:
        x = s
        for layer in self.hidden_layers:
            x = layer(x)
        raw_delta = self.out_layer(x)

        delta = self.epsilon * tf.tanh(raw_delta)
        return delta


def _flatten_states_from_trajectories(trajectories: List[Trajectory]) -> np.ndarray:
    states = []
    for traj in trajectories:
        for step in traj.steps:
            states.append(step.s)
    if not states:
        raise ValueError("No states found in trajectories.")
    return np.stack(states, axis=0).astype(np.float32)


def train_perturbation_net(
    trajectories: List[Trajectory],
    reward_model: RewardModel,   
    victim_policy,               
    target_policy,               
    obs_space: gym.spaces.Box,
    state_dim: int,
    num_actions: int,
    epsilon: float = 0.1,
    lambda_pen: float = 10.0,
    beta: float = 1.0,
    batch_size: int = 64,
    num_epochs: int = 50,
    lr: float = 1e-4,
) -> PerturbationNet:


 
    all_states = _flatten_states_from_trajectories(trajectories)  # [N, state_dim]
    dataset = tf.data.Dataset.from_tensor_slices(all_states)
    dataset = dataset.shuffle(buffer_size=min(10_000, all_states.shape[0])).batch(batch_size)


    v_net = PerturbationNet(state_dim=state_dim, hidden_dims=(128, 128), epsilon=epsilon)
    optimizer = tf.keras.optimizers.Adam(lr)

    obs_min = tf.convert_to_tensor(obs_space.low, dtype=tf.float32)
    obs_max = tf.convert_to_tensor(obs_space.high, dtype=tf.float32)

    for epoch in range(num_epochs):
        for s_batch in dataset:

            with tf.GradientTape() as tape:

                delta = v_net(s_batch) 
                delta = tf.clip_by_value(delta, -epsilon, epsilon)


                s_tilde = tf.clip_by_value(s_batch + delta, obs_min, obs_max)  


                tstep_adv = ts.restart(s_tilde)
                dist_v = victim_policy.distribution(tstep_adv)

                probs_v = dist_v.probs_parameter() 

                B = tf.shape(s_tilde)[0]
                A = num_actions


                s_rep = tf.repeat(s_tilde, repeats=A, axis=0) 
                a_indices = tf.tile(tf.range(A, dtype=tf.int32), [B])  

                r_flat = reward_model(s_rep, a_indices)  
                r_sa = tf.reshape(r_flat, [B, A])        


                exp_r = tf.reduce_sum(probs_v * r_sa, axis=1)  
                J_batch = tf.reduce_mean(exp_r)

                norms = tf.norm(delta, ord=2, axis=1)                
                L_pen_batch = lambda_pen * tf.reduce_mean(tf.nn.relu(norms - epsilon))

                tstep_clean = ts.restart(s_batch)
                dist_theta = target_policy.distribution(tstep_clean)
                probs_theta = dist_theta.probs_parameter()  

                eps = 1e-8
                p_v = tf.clip_by_value(probs_v, eps, 1.0)
                p_theta = tf.clip_by_value(probs_theta, eps, 1.0)


                kl_per_state = tf.reduce_sum(
                    p_v * (tf.math.log(p_v) - tf.math.log(p_theta)),
                    axis=1,
                )  
                L_KL_batch = beta * tf.reduce_mean(kl_per_state)

                loss = -J_batch + L_pen_batch + L_KL_batch

            grads = tape.gradient(loss, v_net.trainable_variables)
            optimizer.apply_gradients(zip(grads, v_net.trainable_variables))

        print(
            f"[V_omega] Epoch {epoch+1}/{num_epochs} | "
            f"loss={loss.numpy():.4f} | J={J_batch.numpy():.4f} | "
            f"L_pen={L_pen_batch.numpy():.4f} | L_KL={L_KL_batch.numpy():.4f}"
        )

    return v_net