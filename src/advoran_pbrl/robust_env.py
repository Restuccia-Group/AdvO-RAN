# advoran_pbrl/robust_env.py
import numpy as np
import tensorflow as tf
import gym

from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from .perturbation_net import PerturbationNet
from .reward_model import RewardModel


class RobustPolicyPyEnv(py_environment.PyEnvironment):


    def __init__(
        self,
        base_env: gym.Env,
        perturbation_net: PerturbationNet,
        reward_model: RewardModel,
        epsilon: float = 0.1,
    ):
        super().__init__()
        self._env = base_env
        self._v_net = perturbation_net
        self._reward_model = reward_model
        self._epsilon = float(epsilon)

    
        self._v_net.trainable = False

 
        obs_space = self._env.observation_space
        assert isinstance(obs_space, gym.spaces.Box), \
            "RobustPolicyPyEnv assumes Box observation space."
        self._obs_shape = obs_space.shape
        self._obs_min = obs_space.low.astype(np.float32)
        self._obs_max = obs_space.high.astype(np.float32)

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=self._obs_shape,
            dtype=np.float32,
            minimum=self._obs_min,
            maximum=self._obs_max,
            name="observation",
        )


        act_space = self._env.action_space
        if isinstance(act_space, gym.spaces.Discrete):
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(),
                dtype=np.int32,
                minimum=0,
                maximum=act_space.n - 1,
                name="action",
            )
            self._discrete_action = True
        elif isinstance(act_space, gym.spaces.Box):
            self._action_spec = array_spec.BoundedArraySpec(
                shape=act_space.shape,
                dtype=np.float32,
                minimum=act_space.low.astype(np.float32),
                maximum=act_space.high.astype(np.float32),
                name="action",
            )
            self._discrete_action = False
        else:
            raise ValueError(f"Unsupported action space for robust env: {act_space}")

        self._clean_state = None   
        self._pert_state = None    
        self._episode_ended = False



    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec



    def _compute_delta(self, s: np.ndarray) -> np.ndarray:

        s_tf = tf.convert_to_tensor(s[None, ...], dtype=tf.float32) 
        delta_tf = self._v_net(s_tf)[0]  
        delta = delta_tf.numpy()
        delta = np.clip(delta, -self._epsilon, self._epsilon)
        return delta.astype(np.float32)

    def _compute_perturbed_state(self, s: np.ndarray) -> np.ndarray:
        delta = self._compute_delta(s)
        s_tilde = np.clip(s + delta, self._obs_min, self._obs_max)
        return s_tilde.astype(np.float32)

    def _compute_reward_hat(self, s_tilde: np.ndarray, a) -> float:

        s_tf = tf.convert_to_tensor(s_tilde[None, :], dtype=tf.float32)
        if self._discrete_action:
            a_tf = tf.convert_to_tensor([int(a)], dtype=tf.int32)
        else:
            raise NotImplementedError("RewardModel currently assumes discrete actions.")
        r_hat = self._reward_model(s_tf, a_tf).numpy()[0]
        return float(r_hat)



    def _reset(self):
        self._episode_ended = False
        reset_out = self._env.reset()
        if isinstance(reset_out, tuple):
            clean_state, _info = reset_out
        else:
            clean_state, _info = reset_out, {}

        self._clean_state = clean_state.astype(np.float32)
        self._pert_state = self._compute_perturbed_state(self._clean_state)
        return ts.restart(self._pert_state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()


        if self._discrete_action:
            a_env = int(action)
        else:
            a_env = np.asarray(action, dtype=np.float32)


        reward = self._compute_reward_hat(self._pert_state, a_env)


        step_out = self._env.step(a_env)
        if len(step_out) == 5:
            next_clean, _, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            next_clean, _, done, info = step_out

        self._clean_state = next_clean.astype(np.float32)
        self._pert_state = self._compute_perturbed_state(self._clean_state)

        if done:
            self._episode_ended = True
            return ts.termination(self._pert_state, reward)
        else:
            return ts.transition(self._pert_state, reward, discount=1.0)


def make_robust_tf_env(
    base_env: gym.Env,
    perturbation_net: PerturbationNet,
    reward_model: RewardModel,
    epsilon: float = 0.1,
):

    py_env = RobustPolicyPyEnv(
        base_env=base_env,
        perturbation_net=perturbation_net,
        reward_model=reward_model,
        epsilon=epsilon,
    )
    return tf_py_environment.TFPyEnvironment(py_env)
