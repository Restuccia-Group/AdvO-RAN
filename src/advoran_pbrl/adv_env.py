import numpy as np
import tensorflow as tf
import gym

from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from .reward_model import RewardModel


class AdvORANAttackPyEnv(py_environment.PyEnvironment):


    def __init__(
        self,
        base_env: gym.Env,        
        victim_policy,            
        reward_model: RewardModel,
        epsilon: float = 0.05,
        lambda_pen: float = 10.0,
    ):
        super().__init__()
        self._env = base_env
        self._victim_policy = victim_policy
        self._reward_model = reward_model
        self._epsilon = float(epsilon)
        self._lambda_pen = float(lambda_pen)

        obs_space = self._env.observation_space


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

        self._action_spec = array_spec.BoundedArraySpec(
            shape=self._obs_shape,
            dtype=np.float32,
            minimum=-self._epsilon,
            maximum=self._epsilon,
            name="delta",
        )

        self._state = None
        self._episode_ended = False



    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def _reset(self):
        self._episode_ended = False
        reset_out = self._env.reset()
        if isinstance(reset_out, tuple):
            obs, _info = reset_out
        else:
            obs, _info = reset_out, {}
        self._state = obs.astype(np.float32)
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        
        delta = np.clip(action, self._action_spec.minimum, self._action_spec.maximum)

        
        s_tilde = np.clip(
            self._state + delta,
            self._observation_spec.minimum,
            self._observation_spec.maximum,
        ).astype(np.float32)

        
        a_v = self._victim_act(s_tilde)


        step_out = self._env.step(a_v)
        if len(step_out) == 5:
            next_state, _, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            next_state, _, done, info = step_out

        next_state = next_state.astype(np.float32)

        
        s_tilde_tf = tf.convert_to_tensor(s_tilde[None, :], dtype=tf.float32)
        a_v_tf = tf.convert_to_tensor([a_v], dtype=tf.int32)
        r_hat = float(self._reward_model(s_tilde_tf, a_v_tf).numpy()[0])

        
        norm_delta = float(np.linalg.norm(delta.astype(np.float32), ord=2))
        penalty = self._lambda_pen * max(0.0, norm_delta - self._epsilon)

        reward = r_hat - penalty

        self._state = next_state
        if done:
            self._episode_ended = True
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward, discount=1.0)

    

    def _victim_act(self, obs: np.ndarray) -> int:
        obs_tf = tf.convert_to_tensor(obs[None, ...], dtype=tf.float32)
        tstep = ts.restart(obs_tf)
        action_step = self._victim_policy.action(tstep)
        action = action_step.action.numpy()[0]
        return int(action)


def make_adv_tf_env(
    base_env: gym.Env,        #
    victim_policy,            
    reward_model: RewardModel,
    epsilon: float = 0.05,
    lambda_pen: float = 10.0,
):
    py_env = AdvORANAttackPyEnv(
        base_env=base_env,
        victim_policy=victim_policy,
        reward_model=reward_model,
        epsilon=epsilon,
        lambda_pen=lambda_pen,
    )
    return tf_py_environment.TFPyEnvironment(py_env)